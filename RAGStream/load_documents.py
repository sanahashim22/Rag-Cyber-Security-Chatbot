"""
load_documents.py
-----------------
Helper module for loading cybersecurity standard documents (PDF and TXT)
into the RAG system.

How to add new content:
  1. Put your PDF or TXT file inside the RAGStream/docs/ folder.
  2. Restart the RAGStream app — it will automatically detect and load all files.
  3. That's it. No code changes needed.

Supported formats: .pdf, .txt
"""

import os
from pathlib import Path
from typing import List

# ── Optional PDF support ─────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


DOCS_FOLDER = Path(__file__).parent / "docs"


def load_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.
    Requires pdfplumber: pip install pdfplumber
    """
    if not PDF_SUPPORT:
        raise ImportError(
            "pdfplumber is not installed. Run: pip install pdfplumber\n"
            "Or with poetry: poetry add pdfplumber"
        )
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)


def load_txt(file_path: str) -> str:
    """Load plain text from a .txt file (UTF-8 encoding)."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_single_file(file_path: str) -> str:
    """
    Load a single file (PDF or TXT) and return its text content.
    Raises ValueError for unsupported file types.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .txt are supported.")


def load_all_from_docs_folder() -> str:
    """
    Scan the docs/ folder inside RAGStream/ and load ALL .pdf and .txt files.
    Returns all content combined into a single string, separated by headers.

    This is how the system auto-loads your cybersecurity documents at startup.
    """
    if not DOCS_FOLDER.exists():
        return ""

    supported_extensions = {".pdf", ".txt"}
    all_text_parts = []

    files_found = sorted([
        f for f in DOCS_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ])

    if not files_found:
        return ""

    for file_path in files_found:
        try:
            content = load_single_file(str(file_path))
            if content.strip():
                # Add a section header so the model knows which document content came from
                header = f"\n\n=== DOCUMENT: {file_path.name} ===\n\n"
                all_text_parts.append(header + content)
        except Exception as e:
            print(f"[load_documents] Warning: Could not load {file_path.name}: {e}")

    return "\n\n".join(all_text_parts)


def load_uploaded_file(uploaded_file) -> str:
    """
    Load content from a Streamlit uploaded file object.
    Supports .pdf and .txt files.
    Used for the file uploader widget in RAGStream.
    """
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    elif file_extension == "pdf":
        if not PDF_SUPPORT:
            raise ImportError(
                "pdfplumber is not installed. Cannot read PDF files.\n"
                "Run: pip install pdfplumber"
            )
        import io
        raw_bytes = uploaded_file.getvalue()
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text.strip())
        return "\n\n".join(text_parts)

    else:
        raise ValueError(f"Unsupported file type: .{file_extension}. Only .pdf and .txt are supported.")


def list_loaded_documents() -> List[str]:
    """Return names of all documents currently in the docs/ folder."""
    if not DOCS_FOLDER.exists():
        return []
    supported_extensions = {".pdf", ".txt"}
    return [
        f.name for f in sorted(DOCS_FOLDER.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
