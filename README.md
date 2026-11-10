# Indian Legal RAG Chatbot (no API keys)

## Files
Add these two PDFs to `data/` with exactly these names:
- `bharatiya-nyaya-sanhita-bns.pdf`
- `Indian cons.pdf`

The app uses ChromaDB's local default ONNX embedding model and the public
SmolLM2-360M-Instruct Q4_K_M GGUF model. No Gemini key, Hugging Face token,
or user API key is needed.

## Run locally (Windows PowerShell)
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

If `llama-cpp-python` fails to build on Windows, install CMake and Visual
Studio Build Tools with the C++ desktop workload, then retry.

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
2.7 GB for ordinary apps). The 360M Q4_K_M model is small (about 271 MB),
but generation can still be slow and the app can exceed free resource limits.
The small model is not dependable for complex legal reasoning. Always verify
important answers against the official PDFs. Chroma's index is built in
temporary storage at startup and rebuilt after a full restart. First startup
downloads the model and local embedding model. PDFs must contain selectable
text; scanned PDFs need OCR first.

If the app exceeds memory, becomes slow, or restarts, free hosting may not be
sufficient for reliable local LLM inference. This package is a best-effort
free deployment, not a guaranteed production legal service.
