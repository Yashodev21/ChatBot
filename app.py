import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

import fitz
import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Indian Legal RAG Chatbot",
    page_icon="⚖️",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PDF_FILES = {
    "bns": DATA_DIR / "bharatiya-nyaya-sanhita-bns.pdf",
    "constitution": DATA_DIR / "Indian cons.pdf",
}

# Stronger than the previous 360M model.
# Downloads on first run; needs more RAM and disk space.
MODEL_REPO = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
TOP_K = 5
MAX_CONTEXT_CHARS = 6500
MAX_ANSWER_TOKENS = 350


# ---------------- PDF EXTRACTION ----------------

def normalize_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text, words_per_chunk=CHUNK_WORDS,
                      overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end]).strip()

        if len(chunk) >= 40:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def get_file_signature():
    digest = hashlib.sha256()

    for path in PDF_FILES.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Required PDF not found: data/{path.name}"
            )

        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def read_pdf_pages(_file_signature):
    # file_signature invalidates cached extraction if a PDF changes.
    del _file_signature
    pages = []

    for doc_type, pdf_path in PDF_FILES.items():
        with fitz.open(pdf_path) as pdf:
            document_pages = 0
            for page_number, page in enumerate(pdf, start=1):
                text = normalize_text(page.get_text("text"))

                if text:
                    document_pages += 1
                    pages.append({
                        "doc_type": doc_type,
                        "document": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    })

            if not document_pages:
                raise ValueError(
                    f"No selectable text was extracted from {pdf_path.name}."
                )

    if not pages:
        raise ValueError(
            "No selectable text was extracted. If the PDFs are scanned, "
            "they must be OCR-processed first."
        )

    return pages


def build_chunks(pages):
    chunks = []

    for page in pages:
        for chunk_number, chunk_text in enumerate(
            split_into_chunks(page["text"])
        ):
            chunks.append({
                "id": (
                    f"{page['doc_type']}-p{page['page']}-"
                    f"c{chunk_number}"
                ),
                "doc_type": page["doc_type"],
                "document": page["document"],
                "page": page["page"],
                "text": chunk_text,
            })

    return chunks


# ---------------- BM25 SEARCH (NO EMBEDDING API) ----------------

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text):
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1
    ]


def source_name(doc_type):
    return "BNS" if doc_type == "bns" else "Constitution"


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(x["text"]) for x in chunks]
        self.doc_lengths = [len(x) for x in self.tokenized]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0
        )
        self.doc_freq = Counter()

        for tokens in self.tokenized:
            self.doc_freq.update(set(tokens))

        self.total_docs = len(self.tokenized)

    def search(self, query, doc_type=None, top_k=TOP_K):
        query_tokens = tokenize(query)

        if not query_tokens or not self.total_docs:
            return []

        results = []

        for index, tokens in enumerate(self.tokenized):
            chunk = self.chunks[index]

            if doc_type and chunk["doc_type"] != doc_type:
                continue

            frequencies = Counter(tokens)
            doc_length = self.doc_lengths[index]
            score = 0.0

            for term in query_tokens:
                frequency = frequencies.get(term, 0)

                if not frequency:
                    continue

                df = self.doc_freq.get(term, 0)
                idf = math.log(
                    1 + (self.total_docs - df + 0.5) / (df + 0.5)
                )

                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * (
                        doc_length / (self.avgdl or 1)
                    )
                )

                score += idf * (
                    frequency * (self.k1 + 1)
                ) / denominator

            if score > 0:
                results.append((score, chunk))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:top_k]]


@st.cache_resource(show_spinner="Preparing legal search...")
def load_search_index(file_signature):
    pages = read_pdf_pages(file_signature)
    chunks = build_chunks(pages)

    if not chunks:
        raise ValueError("The PDFs did not produce searchable text.")

    return pages, chunks, BM25Index(chunks)


# ---------------- LOCAL LANGUAGE MODEL ----------------

@st.cache_resource(show_spinner="Loading the local language model...")
def load_llm():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    cpu_count = os.cpu_count() or 4

    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=max(2, min(4, cpu_count - 1)),
        n_batch=128,
        verbose=False,
    )


# ---------------- ARTICLE / SECTION DETECTION ----------------

def detect_provision(question):
    q = question.lower()

    article_match = re.search(
        r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", q
    )

    if article_match:
        return {
            "kind": "article",
            "number": article_match.group(1),
            "doc_type": "constitution",
        }

    section_match = re.search(
        r"\b(?:bns\s+)?(?:section|sec\.?)\s*(\d+[a-z]?)\b"
        r"|\bbns\s+(\d+[a-z]?)\b",
        q,
    )

    if section_match:
        number = section_match.group(1) or section_match.group(2)
        return {
            "kind": "section",
            "number": number,
            "doc_type": "bns",
        }

    return None


# ---------------- EXACT PROVISION SEARCH ----------------

def find_provision_on_pages(pages, provision):
    number = re.escape(provision["number"])
    doc_type = provision["doc_type"]

    # Accept common legal-PDF heading styles:
    # 21. Protection...
    # 21.—Protection...
    # 21) Protection...
    heading_re = re.compile(
        rf"(?m)^[ \t]*{number}[ \t]*[.)—–-][ \t]*"
        rf"([^\n]{{3,}})"
    )

    alternate_re = re.compile(
        rf"(?m)^[ \t]*{number}[ \t]*[.)—–-]?[ \t]*\n"
        rf"[ \t]*([A-Z][^\n]{{3,}})"
    )

    next_heading_re = re.compile(
        r"(?m)^[ \t]*\d+[ \t]*[.)—–-][ \t]+"
    )
    matches = []

    for page in pages:
        if page["doc_type"] != doc_type:
            continue

        text = page["text"]
        match = heading_re.search(text) or alternate_re.search(text)

        if not match:
            continue

        next_heading = next_heading_re.search(text, match.end())
        end = next_heading.start() if next_heading else len(text)
        extracted = text[match.start():end].strip()

        if len(tokenize(extracted)) < 20:
            continue

        matches.append({
            "text": extracted,
            "doc_type": doc_type,
            "document": page["document"],
            "page": page["page"],
        })

    # Constitution schedules also contain numbered paragraphs. The first
    # matching heading is the article in the main text.
    return matches[:1]


# ---------------- RETRIEVAL ----------------

def retrieve(index, pages, question):
    provision = detect_provision(question)

    if provision:
        exact = find_provision_on_pages(pages, provision)

        if exact:
            # Include the matching page and adjacent pages because
            # long legal provisions can continue onto another page.
            page_numbers = {item["page"] for item in exact}
            selected_pages = [
                page for page in pages
                if page["doc_type"] == provision["doc_type"]
                and any(
                    abs(page["page"] - number) <= 1
                    for number in page_numbers
                )
            ]

            context = "\n\n".join(
                f"[Source: {source_name(page['doc_type'])}, "
                f"page {page['page']}]\n{page['text']}"
                for page in selected_pages
            )

            return context[:MAX_CONTEXT_CHARS]

        # Do not answer a specific provision using an unrelated section.
        return ""

    hits = index.search(question, top_k=TOP_K)

    if not hits:
        return ""

    parts = []
    seen = set()

    for hit in hits:
        normalized = " ".join(hit["text"].lower().split())

        if normalized in seen:
            continue

        seen.add(normalized)
        parts.append(
            f"[Source: {source_name(hit['doc_type'])}, "
            f"page {hit['page']}]\n{hit['text']}"
        )

    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ---------------- ANSWER GENERATION ----------------

def generate_answer(llm, question, context):
    if not context.strip():
        return (
            "I couldn't find sufficient supporting text for this question "
            "in the two supplied PDFs. Please check whether the relevant "
            "provision is present and text-readable."
        )

    system_prompt = """You are a careful Indian legal document assistant.

Answer using ONLY the supplied source excerpts.

Rules:
- Do not use outside knowledge.
- Do not invent articles, sections, definitions, punishments or cases.
- Explain what the supplied text actually says.
- Include the source page number when it is present in the excerpts.
- If the excerpts are incomplete, say what is missing.
- Do not repeat sentences, headings or bullet points.
- Do not write generic filler such as "Important points".
- Use plain, simple English.
- Keep the answer focused and concise.
- This is general legal information, not legal advice.

Write a short paragraph first. Use bullets only when the source
contains several distinct points that need explaining."""

    user_prompt = f"""SOURCE EXCERPTS:
{context}

QUESTION:
{question}

Answer based only on these excerpts."""

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        top_p=0.8,
        repeat_penalty=1.15,
        max_tokens=MAX_ANSWER_TOKENS,
        stream=False,
    )

    answer = (
        result["choices"][0]["message"].get("content") or ""
    ).strip()

    if not answer:
        return (
            "I couldn't generate an answer from the retrieved text. "
            "Please try asking in a different way."
        )

    return answer


# ---------------- LOAD APPLICATION ----------------

try:
    signature = get_file_signature()
    pages, chunks, search_index = load_search_index(signature)
except (FileNotFoundError, ValueError, fitz.FileDataError) as exc:
    print(f"Startup error: {type(exc).__name__}: {exc}")
    st.error(
        "The legal PDFs are missing, empty, unreadable, or contain no "
        "selectable text."
    )
    st.stop()
except (OSError, MemoryError) as exc:
    print(f"Startup error: {type(exc).__name__}: {exc}")
    st.error("The legal documents could not be loaded because of a resource limitation.")
    st.stop()

try:
    llm = load_llm()
except MemoryError as exc:
    print(f"Model error: {type(exc).__name__}: {exc}")
    st.error("The language model needs more memory than this host currently provides.")
    st.stop()
except Exception as exc:
    print(f"Model error: {type(exc).__name__}: {exc}")
    st.error(
        "The language model could not be downloaded or loaded. "
        "Please try again later."
    )
    st.stop()


# ---------------- SIMPLE USER INTERFACE ----------------

st.title("⚖️ Indian Legal RAG Chatbot")

st.caption(
    "Ask questions about the Bharatiya Nyaya Sanhita, 2023 "
    "and the Constitution of India."
)

st.caption("Get clear answers to your Indian law-related questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about Indian law...")

if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the legal documents..."):
                context = retrieve(search_index, pages, question)

            with st.spinner("Preparing your answer..."):
                answer = generate_answer(llm, question, context)

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as exc:
            print(f"Question error: {type(exc).__name__}: {exc}")

            answer = (
                "Sorry, I couldn't process your question. "
                "Please try again."
            )

            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })
