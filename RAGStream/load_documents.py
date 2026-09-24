"""
load_documents.py
-----------------
Helper module for loading cybersecurity standard documents (PDF, TXT, and
JSON rule docs) into the RAG system.

How to add new content:
  1. Put your PDF, TXT, or JSON file inside the RAGStream/docs/ folder.
  2. Restart the RAGStream app — it will automatically detect and load all files.
  3. That's it. No code changes needed.

A JSON file shaped as a dict keyed by sid/id (like
rule_docs_preprocessed_by_sid.json) is formatted as ONE readable block PER
RULE (blank-line separated), each starting with "Snort rule sid <n>:", so
keyword/BM25 search can match on a specific rule's fields. Note: the
combined text from every document in docs/ is then re-chunked by word count
(see chunk_text_with_overlap in RAGStream.py), same as project 2's
rag_engine.py - a 200-word window can still span or split individual rule
blocks, it just won't ever mix raw JSON with no field structure.

Supported formats: .pdf, .txt, .json
"""

import json
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


def _rule_obj_to_text(rule: dict) -> str:
    """Turns one Snort rule's fields into a readable, embeddable block. Keeps
    field names visible (sid:, msg:, classtype:, etc.) so keyword/BM25 search
    can match on them, and puts sid first since that's what alerts get
    cross-checked against."""
    if not isinstance(rule, dict):
        return str(rule)
    ordered_keys = ["sid", "gid", "rev", "msg", "classtype", "action", "protocol",
                     "src_net", "src_port", "dst_net", "dst_port", "flow",
                     "metadata", "content_matches", "rule_category",
                     "rule_text", "doc_url", "direction_label"]
    lines = [f"Snort rule sid {rule.get('sid', '?')}:"]
    seen = set()
    for k in ordered_keys:
        if k in rule and rule[k] not in (None, ""):
            lines.append(f"  {k}: {rule[k]}")
            seen.add(k)
    for k, v in rule.items():
        if k not in seen and v not in (None, ""):
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def load_json(file_path: str) -> str:
    """Converts a JSON file into readable text blocks for embedding.

    Handles two shapes:
      1. A dict keyed by sid/id (e.g. {"105": {...}, "108": {...}, ...},
         like rule_docs_preprocessed_by_sid.json) -> ONE block per rule.
      2. Anything else (a list of objects, or a plain nested dict) -> one
         block per top-level list item, or a single pretty-printed block.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    if isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values()):
        blocks = [_rule_obj_to_text(rule) for rule in data.values()]
        return "\n\n".join(blocks)

    if isinstance(data, list):
        blocks = [_rule_obj_to_text(item) if isinstance(item, dict) else str(item)
                  for item in data]
        return "\n\n".join(blocks)

    return json.dumps(data, indent=2, ensure_ascii=False)


def load_single_file(file_path: str) -> str:
    """
    Load a single file (PDF, TXT, or JSON) and return its text content.
    Raises ValueError for unsupported file types.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    elif ext == ".json":
        return load_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf, .txt, and .json are supported.")


def load_all_from_docs_folder() -> str:
    """
    Scan the docs/ folder inside RAGStream/ and load ALL .pdf, .txt, and
    .json files. Returns all content combined into a single string,
    separated by headers.

    This is how the system auto-loads your cybersecurity documents at startup.
    """
    if not DOCS_FOLDER.exists():
        return ""

    supported_extensions = {".pdf", ".txt", ".json"}
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
    supported_extensions = {".pdf", ".txt", ".json"}
    return [
        f.name for f in sorted(DOCS_FOLDER.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
