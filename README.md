# Indian Legal RAG Chatbot (no API keys)

## Files
Add these two PDFs to `data/` with exactly these names:
- `bharatiya-nyaya-sanhita-bns.pdf`
- `Indian cons.pdf`

The app uses BM25 keyword retrieval and the public
Qwen2.5-1.5B-Instruct Q4_K_M GGUF model. No Gemini key, Hugging Face token,
or user API key is needed. ChromaDB and ONNX are not used.

## Run locally (Windows PowerShell)
    py -3.9 -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

The application code remains compatible with Python 3.9.6. The pinned
dependencies are selected for compatibility with both Python 3.9 and 3.11.
The deployment environment should remain on Python 3.11 because that is the
currently configured Streamlit Community Cloud runtime; changing it is not
required by the application.

The deployment requirements use the public CPU-wheel index for
`llama-cpp-python`, so Streamlit Community Cloud does not need to compile it
from source. Local Windows installation may still need a compiler if the
selected wheel is not available for the local interpreter.

## Deploy free on Streamlit Community Cloud
1. Push this whole project to a GitHub repository.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Click **Create app**.
4. Select your repository, `main` branch, and `app.py`.
5. In Advanced settings, select Python 3.11.
6. Click Deploy and wait for model/dependency downloads.
7. Share the resulting public `https://....streamlit.app` URL.

No Streamlit secrets are required.

## Git commands
    git init
    git add .
    git commit -m "Create local legal RAG chatbot"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/legal-rag-streamlit.git
    git push -u origin main

## Free-hosting caveats
Streamlit Community Cloud has limited CPU/RAM (documented maximum around
2.7 GB for ordinary apps). The 1.5B Q4_K_M model is larger than the former
360M model, so generation can still be slow or exceed free resource limits.
The model is cached by Streamlit and Hugging Face after the first successful
download. Always verify important answers against the official PDFs. PDFs
must contain selectable text; scanned PDFs need OCR first.

If the app exceeds memory, becomes slow, or restarts, free hosting may not be
sufficient for reliable local LLM inference. This package is a best-effort
free deployment, not a guaranteed production legal service.
