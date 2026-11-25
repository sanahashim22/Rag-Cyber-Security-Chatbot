"""Process-wide singletons built once at FastAPI startup - the equivalent
of dashboard_app.py's @st.cache_resource load_rag_system(), just without a
Streamlit decorator to do the "build once" part (main.py's startup event
does it instead)."""

from __future__ import annotations

from typing import Any, List, Optional

rag_system: Optional[Any] = None
doc_names: List[str] = []


def build_rag_system():
    """Builds the shared ConvoRAG instance from whatever's in docs/ (NIST/
    ISO text), same chunking/embedding config dashboard_app.py uses."""
    global rag_system, doc_names

    import rag_core
    from load_documents import load_all_from_docs_folder, list_loaded_documents
    from . import ranking_service

    docs_folder_text = load_all_from_docs_folder()
    names = list_loaded_documents()
    if docs_folder_text.strip():
        document_text = docs_folder_text
    else:
        document_text = rag_core.DEFAULT_CYBERSECURITY_INFO
        names = []

    documents = rag_core.chunk_text_with_overlap(text=document_text, chunk_size=200, overlap_size=40)
    search_mode = "hybrid" if rag_core.ADVANCED_MODE else "vector"
    rag_system = rag_core.ConvoRAG(
        documents,
        embedding_model="nomic-embed-text",
        llm_model=ranking_service.RANKING_MODELS[0],
        search_mode=search_mode,
    )
    doc_names = names
    return rag_system, doc_names
