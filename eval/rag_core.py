"""
rag_core.py
-----------
Headless (terminal-friendly) copy of the ConvoRAG class from RAGStream.py.

WHY THIS FILE EXISTS:
RAGStream.py's ConvoRAG class calls st.write / st.spinner / st.error
everywhere, so it can't be imported and run outside a Streamlit app.
This file is the SAME retrieval + generation logic (search, embed_text,
cosine_similarity, detect_query_type, contextualize_query, rag), just
with every `st.*` call swapped for a plain print() (or removed).

Nothing about the actual RAG behaviour (chunking, hybrid search, prompts,
thresholds) has been changed. If you tweak RAGStream.py later, mirror the
change here too, or your eval results won't reflect the real app.
"""

import heapq
import re
import hashlib
import time
from typing import List, Tuple

import numpy as np
import ollama

try:
    import chromadb
    from rank_bm25 import BM25Okapi
    ADVANCED_MODE = True
except ImportError:
    ADVANCED_MODE = False


def extract_identifier_terms(text: str) -> List[str]:
    return sorted(set(re.findall(r'\b[a-z]{2}(?:\.[a-z0-9]{2,4})+\b', text.lower())))


def acronym_boost(query: str, document: str) -> float:
    query_terms = extract_identifier_terms(query)
    if not query_terms:
        return 0.0
    document_lower = document.lower()
    document_compact = re.sub(r'[^a-z0-9]+', '', document_lower)
    boost = 0.0
    for term in query_terms:
        term_compact = re.sub(r'[^a-z0-9]+', '', term)
        if term in document_lower or term_compact in document_compact:
            boost += 0.8
    return min(boost, 1.0)


def chunk_text_with_overlap(text: str, chunk_size: int = 200, overlap_size: int = 40) -> List[str]:
    """Identical logic to RAGStream.chunk_text_with_overlap (st.write calls removed)."""
    if not text:
        return ["No text provided"]

    words = text.split()
    if not words:
        return ["No text provided"]
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    i = 0
    while i < len(words):
        chunk_end = min(i + chunk_size, len(words))
        chunk = " ".join(words[i:chunk_end])
        chunks.append(chunk)
        i += chunk_size - overlap_size
        if i + chunk_size > len(words) and i < len(words):
            remaining = len(words) - i
            if remaining > overlap_size:
                chunks.append(" ".join(words[i:]))
            break

    improved_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > (chunk_size / 2):
            for marker in ["\n\n", "\n", ". ", "! ", "? "]:
                if marker in chunk:
                    last_marker_pos = chunk.rfind(marker)
                    if last_marker_pos > int(len(chunk) - overlap_size):
                        improved_chunks.append(chunk[: last_marker_pos + len(marker)])
                        break
            else:
                improved_chunks.append(chunk)
        else:
            improved_chunks.append(chunk)

    return improved_chunks if improved_chunks else chunks


class ConvoRAGHeadless:
    """Same behaviour as RAGStream.ConvoRAG, but prints instead of using st.*,
    and has no Streamlit session-state dependency."""

    def __init__(self, documents: List[str], embedding_model: str = "nomic-embed-text",
                 llm_model: str = "llama3.2", verbose: bool = True):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.advanced_mode = ADVANCED_MODE
        self.verbose = verbose
        self.document_embeddings = []
        self.conversation_history = []

        if self.verbose:
            mode_str = "Advanced (Hybrid + DB)" if self.advanced_mode else "Basic (In-Memory)"
            print(f"[rag_core] {len(documents)} chunks | mode={mode_str} | embed={embedding_model} | llm={llm_model}")

        if self.advanced_mode:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="cyber_standards", metadata={"hnsw:space": "cosine"}
            )
            full_text = "".join(self.documents)
            doc_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()
            existing_count = self.collection.count()
            existing_meta = self.collection.metadata or {}
            stored_hash = existing_meta.get("doc_hash", "")
            needs_rebuild = (existing_count == 0 or existing_count != len(self.documents) or stored_hash != doc_hash)

            if needs_rebuild:
                if self.verbose:
                    print("[rag_core] (Re)building embeddings in ChromaDB... this happens once, not per model.")
                try:
                    self.chroma_client.delete_collection(name="cyber_standards")
                except Exception:
                    pass
                self.collection = self.chroma_client.create_collection(
                    name="cyber_standards", metadata={"hnsw:space": "cosine", "doc_hash": doc_hash}
                )
                for i, doc in enumerate(documents):
                    try:
                        response = ollama.embeddings(model=self.embedding_model, prompt=doc)
                        self.collection.add(
                            embeddings=[response["embedding"]], documents=[doc],
                            metadatas=[{"chunk_id": i}], ids=[f"chunk_{doc_hash}_{i}"]
                        )
                    except Exception as e:
                        print(f"[rag_core] Error embedding chunk {i}: {e}")
            elif self.verbose:
                print(f"[rag_core] Reusing {existing_count} existing chunks already in chroma_db.")

            def tokenize(text):
                return re.findall(r'[a-z0-9.-]+', text.lower())
            self.tokenize = tokenize
            tokenized_corpus = [self.tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            if self.verbose:
                print("[rag_core] Embedding documents in memory (basic mode)...")
            self.document_embeddings = [self.embed_text(doc) for doc in documents]

    def embed_text(self, text: str) -> np.ndarray:
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return np.array(response["embedding"])
        except Exception as e:
            print(f"[rag_core] Embedding error: {e}")
            return np.zeros(768)

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"[rag_core] Cosine similarity error: {e}")
            return 0.0

    def topk(self, arr: List[float], k: int) -> List[int]:
        if not arr:
            return []
        k = min(k, len(arr))
        try:
            return heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
        except Exception as e:
            print(f"[rag_core] topk error: {e}")
            return list(range(min(k, len(arr))))

    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        if not self.documents:
            return "No documents available.", 0.0
        try:
            if self.advanced_mode:
                query_embedding = self.embed_text(query)
                keyword_boosts = [acronym_boost(query, doc) for doc in self.documents]

                vector_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()], n_results=top_k
                )
                vector_indices = [meta["chunk_id"] for meta in vector_results["metadatas"][0]] if vector_results["metadatas"] else []

                tokenized_query = self.tokenize(query)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_indices = self.topk(bm25_scores.tolist(), top_k)

                k_rrf = 60
                combined_scores = {}
                for rank, idx in enumerate(vector_indices):
                    combined_scores[idx] = combined_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
                for rank, idx in enumerate(bm25_indices):
                    combined_scores[idx] = combined_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
                for idx, boost in enumerate(keyword_boosts):
                    if boost > 0:
                        combined_scores[idx] = combined_scores.get(idx, 0.0) + boost

                sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
                if not sorted_indices:
                    return "No relevant documents found.", 0.0

                result = "\n".join([self.documents[i] for i in sorted_indices])
                top_rrf_score = combined_scores[sorted_indices[0]]
                return result, top_rrf_score
            else:
                if not self.document_embeddings:
                    return "No documents available.", 0.0
                query_embedding = self.embed_text(query)
                boosted_scores = []
                for idx, doc_embedding in enumerate(self.document_embeddings):
                    similarity = self.cosine_similarity(query_embedding, doc_embedding)
                    boosted_scores.append(min(1.0, similarity + acronym_boost(query, self.documents[idx])))
                if not boosted_scores:
                    return "No similarities found.", 0.0
                topk_indices = self.topk(boosted_scores, top_k)
                if not topk_indices:
                    return "No relevant documents found.", 0.0
                result = "\n".join([self.documents[i] for i in topk_indices])
                return result, boosted_scores[topk_indices[0]]
        except Exception as e:
            print(f"[rag_core] Search error: {e}")
            return "Error occurred while searching documents.", 0.0

    def generate_answer(self, system_prompt: str, user_prompt: str, inject_history: bool = False) -> str:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if inject_history and self.conversation_history:
                for (q, a) in self.conversation_history[-3:]:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": user_prompt})

            response = ollama.chat(model=self.llm_model, messages=messages, stream=False)
            return response["message"]["content"]
        except Exception as e:
            return f"Error generating response: {e}"

    def detect_query_type(self, query: str, relevance_threshold: float = 0.03) -> str:
        system_prompt = """
        You are an expert at classifying cybersecurity and information security queries. Your task is to categorize each query into EXACTLY ONE of these categories:

        1. "cybersecurity-related" - Questions about cybersecurity standards, frameworks, controls, policies, risk management, compliance, threats, vulnerabilities, security functions, governance, etc.
        2. "compliment" - Positive feedback or appreciation
        3. "complaint" - Negative feedback or dissatisfaction
        4. "chitchat" - General conversation not directly related to cybersecurity
        5. "off-topic" - Questions completely unrelated to cybersecurity or information security

        IMPORTANT RULES:
        - Follow-up questions about cybersecurity topics should be classified as "cybersecurity-related" even if they are brief
        - Treat ambiguous queries that could reasonably be about cybersecurity as "cybersecurity-related"
        - WHEN IN DOUBT, classify as "cybersecurity-related"

        Return ONLY the category name, with no explanation.
        """
        user_prompt = f'Classify this query: "{query}"'
        try:
            initial_classification = self.generate_answer(system_prompt, user_prompt).lower().strip()
            if "cybersecurity-related" in initial_classification:
                classification = "cybersecurity-related"
            elif "off-topic" in initial_classification or "offtopic" in initial_classification:
                classification = "off-topic"
            elif "compliment" in initial_classification:
                classification = "compliment"
            elif "complaint" in initial_classification:
                classification = "complaint"
            elif "chitchat" in initial_classification:
                classification = "chitchat"
            elif "cybersecurity" in initial_classification:
                classification = "cybersecurity-related"
            else:
                classification = "cybersecurity-related"

            if classification == "cybersecurity-related":
                return classification

            if classification in ["chitchat", "off-topic"]:
                _, top_similarity = self.search(query, top_k=1)
                if top_similarity >= relevance_threshold:
                    return "cybersecurity-related"
            return classification
        except Exception as e:
            print(f"[rag_core] Classification error: {e}")
            return "cybersecurity-related"

    def contextualize_query(self, current_query: str) -> str:
        if not self.conversation_history:
            return current_query
        if len(current_query.split()) > 10:
            return current_query

        history_context = "The conversation history is ordered from oldest to most recent:\n\n"
        for idx, (q, a) in enumerate(self.conversation_history[-5:]):
            history_context += f"Exchange {idx + 1}:\nUser: {q}\nAssistant: {a}\n\n"

        system_prompt = """You are a query reformulation system for a cybersecurity standards assistant. Your job is to take a user's query and reformulate it to be self-sufficient by incorporating relevant context from the conversation history, while preserving the original intent.

RULES:
1. Create a single, concise sentence that captures both the current query and relevant context
2. Never add your own opinions, reasoning, explanations, or commentary
3. Preserve all details from the user's current query
4. Be sensitive to topic changes - when a topic changes, do not carry over unrelated context
5. Never answer the question - only reformulate it
6. If the query is already self-sufficient, make minimal changes or return it as is
7. If the user's intent is a question, always ensure the reformulated text is also a question

DO NOT add commentary or explanations. DO NOT exceed one sentence unless absolutely necessary."""

        user_prompt = f'Conversation history:\n{history_context}\n\nCurrent query: "{current_query}"\n\nReformulated query:\n'
        try:
            reformulated_query = self.generate_answer(system_prompt, user_prompt).strip()
            if ":" in reformulated_query:
                parts = reformulated_query.split(":", 1)
                if len(parts) > 1:
                    reformulated_query = parts[1].strip()
            reformulated_query = reformulated_query.strip("\"'")

            problematic_phrases = ["based on", "according to", "from our conversation", "as mentioned", "earlier you asked", "you asked about"]
            if (len(reformulated_query.split()) > 25
                    or any(p in reformulated_query.lower() for p in problematic_phrases)
                    or ("." in reformulated_query and "?" not in reformulated_query)):
                return current_query
            if reformulated_query.lower() == current_query.lower():
                return current_query
            return reformulated_query
        except Exception as e:
            print(f"[rag_core] Contextualize error: {e}")
            return current_query

    def handle_non_cybersecurity_query(self, query_type: str, query: str) -> str:
        prompts = {
            "compliment": "You are an AI assistant for a Cybersecurity Standards information system. A user has given you a compliment or thanked you.\nRespond with a brief, gracious acknowledgment and an offer to help with cybersecurity questions. 1-2 sentences max.",
            "complaint": "You are an AI assistant for a Cybersecurity Standards information system. A user has complained.\nRespond with a brief, professional apology and an offer to help with specific cybersecurity questions. 1-2 sentences max.",
            "chitchat": "You are an AI assistant for a Cybersecurity Standards information system. A user has engaged in general conversation not related to cybersecurity.\nRespond with a brief, friendly response and a polite redirection to cybersecurity topics. 1-2 sentences max.",
        }
        system_prompt = prompts.get(query_type, "You are an AI assistant for a Cybersecurity Standards information system. A user has asked about a topic completely unrelated to cybersecurity.\nRespond with a polite explanation that you assist with cybersecurity standards and frameworks such as NIST CSF 2.0, and offer to help. 1-2 sentences max.")
        user_prompt = f'User message: "{query}"'
        return self.generate_answer(system_prompt, user_prompt)

    def rag(self, query: str) -> str:
        """Main entry point — identical decision flow to RAGStream.ConvoRAG.rag()."""
        try:
            query_type = self.detect_query_type(query)

            if query_type != "cybersecurity-related":
                response = self.handle_non_cybersecurity_query(query_type, query)
                self.conversation_history.append((query, response))
                return response

            contextualized_query = self.contextualize_query(query)
            context, relevance_score = self.search(contextualized_query)

            RELEVANCE_THRESHOLD = 0.02
            if relevance_score < RELEVANCE_THRESHOLD and context != "No documents available.":
                system_prompt = """
                You are an AI assistant specializing in cybersecurity standards and frameworks. You are designed to answer questions about frameworks such as NIST CSF 2.0, ISO 27001, and other cybersecurity standards.

                The user has asked a question that does not match the available document content. Respond with:
                1. A polite explanation that you do not have enough information in your current documents to answer their specific question
                2. An offer to help with other cybersecurity standards questions
                3. A suggestion to check the original standard documents for more specific information

                Keep your response brief and professional.
                """
                user_prompt = f'User question without matching context: "{query}"'
                response = self.generate_answer(system_prompt, user_prompt)
                self.conversation_history.append((query, response))
                return response

            system_prompt = """You are an expert AI assistant specializing in cybersecurity standards and frameworks. You have access to a collection of detailed information from cybersecurity standards documents such as the NIST Cybersecurity Framework (CSF) 2.0 and other official cybersecurity standards.

            Your task is to help users by providing accurate, clear, and well-structured answers about cybersecurity standards, controls, functions, categories, and best practices.

            If a user's question cannot be answered based on the provided context, respond with:
            "I cannot answer this based on the available documents. Please refer to the official standard documentation or consult your cybersecurity administrator for more information."

            Respond in a clear, structured, and professional manner. Use bullet points or numbered lists when explaining complex topics. Always base your answers on the provided document context.
            """
            user_prompt = f'Based on the following context from cybersecurity standards documents, please answer the question.\nIf the answer cannot be derived from the context, say "I cannot answer this based on the available documents." \n\nContext:\n{context}\n\nQuestion: {contextualized_query}\n\nAnswer:\n'

            response = self.generate_answer(system_prompt, user_prompt, inject_history=True)
            self.conversation_history.append((query, response))
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {e}"
