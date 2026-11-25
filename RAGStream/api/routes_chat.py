from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import state
from .schemas import ChatMessageRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/message")
def send_message(req: ChatMessageRequest):
    """Same non-streaming behavior as dashboard_app.py's _draw_chatbot:
    consumes the whole stream server-side and returns the full answer."""
    if state.rag_system is None:
        raise HTTPException(503, "RAG system is still starting up - try again shortly.")

    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question is empty")

    state.rag_system.llm_model = req.model
    try:
        stream = state.rag_system.rag(question)
        # Ollama's streaming chunks are pydantic ChatResponse objects, not
        # plain dicts, but support dict-style .get() same as a real dict -
        # see dashboard_app.py's _draw_chatbot for the bug this avoids.
        answer = "".join(
            (chunk.get("message", {}) or {}).get("content", "") or ""
            for chunk in stream
        )
    except Exception as e:
        answer = f"Sorry, I hit an error answering that: {e}"

    state.rag_system.conversation_history.append((question, answer))
    return {"question": question, "answer": answer}


@router.get("/history")
def get_history():
    if state.rag_system is None:
        return {"history": []}
    return {
        "history": [
            {"question": q, "answer": a} for q, a in state.rag_system.conversation_history
        ]
    }


@router.delete("/history")
def clear_history():
    if state.rag_system is not None:
        state.rag_system.conversation_history = []
    return {"ok": True}
