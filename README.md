# Indian Legal RAG Chatbot

This Streamlit app searches only the two supplied legal PDFs:

- `data/bharatiya-nyaya-sanhita-bns.pdf`
- `data/Indian cons.pdf`

It uses PyMuPDF for extraction and a lightweight in-process BM25 index for
retrieval. Answers are extractive quotations from retrieved PDF text, with
document and page references. It does not use external legal knowledge or
remote inference services.

## Local setup

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 is the deployment target. The application code is also compatible
with Python 3.9.6.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create an app at https://share.streamlit.io/.
3. Select the repository, `main` branch, and `app.py`.
4. Select Python 3.11 in Advanced settings.
5. Deploy.

No secrets, API keys, system packages, model downloads, or external services
are required. The PDFs must contain selectable text; scanned PDFs require OCR
before they can be searched.

## Git cleanup

Do not commit virtual environments, caches, generated files, local databases,
or local binary assets. The repository `.gitignore` covers these files.
