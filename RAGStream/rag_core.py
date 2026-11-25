"""
rag_core.py
-----------
Streamlit-free fork of RAGStream.py's ConvoRAG + chunk_text_with_overlap,
for use by the FastAPI backend (api/).

RAGStream.py's ConvoRAG calls st.write()/st.error()/st.spinner()/
st.expander() directly inside its core methods (search(), rag(),
classify_query(), contextualize_query(), ...) - those assume a live
Streamlit ScriptRunContext and have no place in a plain backend process.
This file is the same logic with every one of those calls replaced by
Python's logging module (same debug detail, no UI coupling), so it can be
imported by api/main.py without ever touching Streamlit.

RAGStream.py itself is UNCHANGED and still used by dashboard_app.py -
this is an additive fork, not a replacement. If RAGStream.py's ConvoRAG
logic ever changes, re-port the change here too.
"""

from __future__ import annotations

import heapq
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np
import ollama

logger = logging.getLogger("rag_core")

# Same env-var convention as RAGStream.py - see its own comment for why this
# is deliberately NOT named OLLAMA_HOST.
RAG_OLLAMA_HOST = os.environ.get("RAG_OLLAMA_HOST")
OLLAMA_HOST = RAG_OLLAMA_HOST or "http://localhost:11434"

ollama_client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"},
    timeout=180,
)

try:
    import chromadb
    from rank_bm25 import BM25Okapi
    ADVANCED_MODE = True
except ImportError:
    ADVANCED_MODE = False

EMBED_BATCH_SIZE = 32
MAX_EMBED_CHARS = 3000


def extract_identifier_terms(text: str) -> List[str]:
    """Return dotted cybersecurity identifiers such as PR.DS or GV.RM."""
    return sorted(set(re.findall(r'\b[a-z]{2}(?:\.[a-z0-9]{2,4})+\b', text.lower())))


def acronym_boost(query: str, document: str) -> float:
    """Boost documents that contain exact identifier-style matches from the query."""
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


DEFAULT_CYBERSECURITY_INFO = """The NIST Cybersecurity Framework (CSF) 2.0 provides guidance to industry, government agencies, and other organizations to manage cybersecurity risks.
The CSF 2.0 is organized around six core Functions: Govern, Identify, Protect, Detect, Respond, and Recover.
The Govern (GV) Function establishes and monitors the organization's cybersecurity risk management strategy, expectations, and policy.
The Identify (ID) Function helps the organization understand its current cybersecurity risks to systems, people, assets, data, and capabilities.
The Protect (PR) Function uses safeguards to prevent or reduce cybersecurity risks including access control, data security, and training.
The Detect (DE) Function finds and analyzes possible cybersecurity attacks and compromises through continuous monitoring.
The Respond (RS) Function takes action regarding a detected cybersecurity incident including incident management and communication.
The Recover (RC) Function restores assets and operations that were impacted by a cybersecurity incident including recovery planning and improvements.
The CSF 2.0 introduced the Govern function as a new addition over CSF 1.1, reflecting the importance of cybersecurity governance at the executive and board level.
Organizational Context (GV.OC) ensures that the organizational mission is understood and that cybersecurity risk management supports and shapes the organization's objectives.
Risk Management Strategy (GV.RM) ensures that the organization's priorities, constraints, risk tolerance, and assumptions are established, communicated, and used to support cybersecurity risk decisions.
Asset Management (ID.AM) involves identifying and managing assets (data, hardware, software, systems, and facilities) that enable the organization to achieve business purposes.
Risk Assessment (ID.RA) identifies and prioritizes risks to the organization's operations, assets, and individuals based on threat and vulnerability information.
Identity Management and Access Control (PR.AA) ensures that access to physical and logical assets is limited to authorized users, processes, and devices.
Awareness and Training (PR.AT) ensures that personnel are provided cybersecurity awareness education and trained to perform their cybersecurity-related duties.
Data Security (PR.DS) ensures that data-at-rest, in-transit, and in-use are managed in accordance with the organization's risk strategy to protect confidentiality, integrity, and availability.
Platform Security (PR.PS) manages the hardware, software, and services of physical and virtual platforms according to the organization's risk strategy to protect confidentiality, integrity, and availability.
Technology Infrastructure Resilience (PR.IR) ensures that security architectures are managed with the organization's risk strategy to protect asset confidentiality, integrity, and availability.
Adverse Event Analysis (DE.AE) ensures that anomalies, indicators of compromise, and other potentially adverse events are analyzed to characterize the events and detect cybersecurity incidents.
Incident Management (RS.MA) covers the response to detected cybersecurity incidents including the execution of the response plan and activities to minimize the impact.
Incident Analysis (RS.AN) ensures that investigations are conducted to ensure effective response and support forensics and recovery activities.
Incident Response Reporting and Communication (RS.CO) covers response activities that are coordinated with internal and external stakeholders as required by laws, regulations, or policies.
Incident Mitigation (RS.MI) ensures that activities are performed to prevent expansion of an event and mitigate its effects.
Incident Recovery Plan Execution (RC.RP) ensures that restoration activities are performed to ensure operational availability of systems and services affected by cybersecurity incidents.
Incident Recovery Communication (RC.CO) ensures that restoration activities are coordinated with internal and external parties such as coordinating centers, Internet Service Providers, vendors, and other CSIRTs.
The CSF Core provides a set of cybersecurity outcomes organized as Functions, Categories, and Subcategories that can be customized for any organization.
CSF Profiles represent the alignment of the CSF Core to an organization's requirements, risk tolerance, and resources to support prioritization of cybersecurity activities.
CSF Tiers describe the maturity of an organization's cybersecurity risk governance and management practices ranging from Tier 1 (Partial) to Tier 4 (Adaptive).
Supply chain risk management is addressed under the Govern function to ensure cybersecurity risks in the supply chain are identified, assessed, and managed."""


class ConvoRAG:
    def __init__(
        self,
        documents: List[str],
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
        search_mode: str = "hybrid",
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.advanced_mode = ADVANCED_MODE
        self.search_mode = search_mode

        mode_str = "Advanced (Hybrid + DB)" if self.advanced_mode else "Basic (In-Memory)"
        logger.debug(f"Initializing with {len(documents)} document chunks in {mode_str} mode")
        logger.debug(f"Using embedding model: {embedding_model}, LLM model: {llm_model}")

        self.document_embeddings = []

        if self.advanced_mode:
            # --- ADVANCED MODE (ChromaDB + BM25) ---
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="cyber_standards",
                metadata={"hnsw:space": "cosine"}
            )

            full_text = "".join(self.documents)
            import hashlib
            doc_hash = hashlib.md5(full_text.encode('utf-8')).hexdigest()

            existing_count = self.collection.count()
            existing_meta = self.collection.metadata or {}
            stored_hash = existing_meta.get("doc_hash", "")

            logger.debug(f"Found {existing_count} existing chunks in the database.")
            logger.debug(f"Stored hash: {stored_hash} | Current hash: {doc_hash}")

            needs_rebuild = (existing_count == 0 or existing_count != len(self.documents) or stored_hash != doc_hash)

            if needs_rebuild:
                logger.debug(f"Generating and storing new embeddings for {len(documents)} chunks (batched)...")
                try:
                    self.chroma_client.delete_collection(name="cyber_standards")
                except Exception:
                    pass

                self.collection = self.chroma_client.create_collection(
                    name="cyber_standards",
                    metadata={"hnsw:space": "cosine", "doc_hash": doc_hash}
                )

                indexed_documents = list(enumerate(documents))
                batches = [
                    indexed_documents[start:start + EMBED_BATCH_SIZE]
                    for start in range(0, len(documents), EMBED_BATCH_SIZE)
                ]

                def embed_batch(batch):
                    indices = [i for i, _ in batch]
                    full_texts = [doc for _, doc in batch]
                    embed_inputs = [doc[:MAX_EMBED_CHARS] for _, doc in batch]
                    errors = []
                    try:
                        response = ollama_client.embed(model=self.embedding_model, input=embed_inputs)
                        embeddings = response["embeddings"]
                        if len(embeddings) != len(embed_inputs):
                            raise ValueError("Ollama returned the wrong number of embeddings")
                        return list(zip(indices, full_texts, embeddings)), errors
                    except Exception:
                        results = []
                        for idx, full_text_, embed_text in zip(indices, full_texts, embed_inputs):
                            try:
                                resp = ollama_client.embeddings(model=self.embedding_model, prompt=embed_text)
                                results.append((idx, full_text_, resp["embedding"]))
                            except Exception as e:
                                errors.append((idx, str(e)))
                        return results, errors

                completed = 0
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for batch_result, batch_errors in executor.map(embed_batch, batches):
                        if batch_result:
                            self.collection.add(
                                embeddings=[e for _, _, e in batch_result],
                                documents=[d for _, d, _ in batch_result],
                                metadatas=[{"chunk_id": i} for i, _, _ in batch_result],
                                ids=[f"chunk_{doc_hash}_{i}" for i, _, _ in batch_result],
                            )
                        for idx, err_msg in batch_errors:
                            logger.error(f"Error embedding chunk {idx}: {err_msg}")
                        completed += EMBED_BATCH_SIZE
                        logger.debug(f"Embedded ~{min(completed, len(documents))}/{len(documents)} chunks...")

            def tokenize(text):
                return re.findall(r'[a-z0-9.-]+', text.lower())
            self.tokenize = tokenize

            tokenized_corpus = [self.tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.debug("Hybrid Search (BM25 + Vector) ready.")

        else:
            # --- BASIC MODE (NumPy In-Memory) ---
            self.document_embeddings = [self.embed_text(doc) for doc in documents]

        self.conversation_history = []

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Ollama's specified embedding model."""
        try:
            response = ollama_client.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            embedding_array = np.array(embedding)
            logger.debug(f"Embedded text '{text[:50]}...' -> {embedding_array.shape} dimensions")
            return embedding_array
        except Exception as e:
            logger.error(f"Error in embedding: {str(e)}")
            return np.zeros(768)

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error in cosine similarity calculation: {str(e)}")
            return 0.0

    def topk(self, arr: List[float], k: int) -> List[int]:
        """Get the k largest elements and their indexes."""
        if not arr:
            return []

        k = min(k, len(arr))

        try:
            topk_indices = heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
            top_scores = [arr[i] for i in topk_indices]
            logger.debug(f"Top {k} similarity scores: {[f'{score:.4f}' for score in top_scores]}")
            return topk_indices
        except Exception as e:
            logger.error(f"Error in topk calculation: {str(e)}")
            return list(range(min(k, len(arr))))

    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Search for the most relevant documents based on cosine similarity of embeddings."""
        if not self.documents:
            return "No documents available.", 0.0

        try:
            logger.debug(f"Searching for: '{query}'")

            if self.advanced_mode and self.search_mode == "vector":
                query_embedding = self.embed_text(query)

                vector_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )

                vector_indices = [meta["chunk_id"] for meta in vector_results["metadatas"][0]] if vector_results["metadatas"] else []
                vector_scores = [1.0 - d for d in vector_results["distances"][0]] if vector_results["distances"] else []

                if not vector_indices:
                    return "No relevant documents found.", 0.0

                result = "\n\n<<<CHUNK_BREAK>>>\n\n".join([self.documents[i] for i in vector_indices])
                top_score = vector_scores[0] if vector_scores else 0.0

                logger.debug(f"Normal (semantic-only) search results. Top vector similarity score: {top_score:.4f}")
                return result, top_score

            elif self.advanced_mode:
                query_embedding = self.embed_text(query)
                keyword_boosts = [acronym_boost(query, doc) for doc in self.documents]

                vector_results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )

                vector_indices = [meta["chunk_id"] for meta in vector_results["metadatas"][0]] if vector_results["metadatas"] else []
                vector_scores = [1.0 - d for d in vector_results["distances"][0]] if vector_results["distances"] else []

                tokenized_query = self.tokenize(query)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_indices = self.topk(bm25_scores.tolist(), top_k)

                k_rrf = 60
                combined_scores: Dict[int, float] = {}

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

                result = "\n\n<<<CHUNK_BREAK>>>\n\n".join([self.documents[i] for i in sorted_indices])

                best_vector = vector_scores[0] if vector_scores else 0.0
                max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
                best_keyword = max(keyword_boosts) if keyword_boosts else 0.0
                top_rrf_score = combined_scores[sorted_indices[0]]

                logger.debug(
                    f"Hybrid search results. Vector: {best_vector:.4f}, Max BM25: {max_bm25:.4f}, "
                    f"Keyword boost: {best_keyword:.4f}, Top RRF combined score: {top_rrf_score:.4f}"
                )
                return result, top_rrf_score

            else:
                if not self.document_embeddings:
                    return "No documents available.", 0.0

                query_embedding = self.embed_text(query)

                similarities = []
                boosted_scores = []
                for idx, doc_embedding in enumerate(self.document_embeddings):
                    similarity = self.cosine_similarity(query_embedding, doc_embedding)
                    similarities.append(similarity)
                    boosted_scores.append(min(1.0, similarity + acronym_boost(query, self.documents[idx])))

                if not boosted_scores:
                    return "No similarities found.", 0.0

                topk_indices = self.topk(boosted_scores, top_k)

                if not topk_indices:
                    return "No relevant documents found.", 0.0

                result = "\n\n<<<CHUNK_BREAK>>>\n\n".join([self.documents[i] for i in topk_indices])
                return result, boosted_scores[topk_indices[0]] if topk_indices else 0.0

        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return "Error occurred while searching documents.", 0.0

    def generate_answer(
        self, system_prompt: str, user_prompt: str, is_stream: bool = True,
        inject_history: bool = False
    ) -> Any:
        """Generate a response using the provided system and user prompts."""
        try:
            logger.debug(f"Generating response with model: {self.llm_model}")

            generation_start = time.time()

            messages = [{"role": "system", "content": system_prompt}]
            if inject_history and self.conversation_history:
                for (q, a) in self.conversation_history[-3:]:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": user_prompt})

            response = ollama_client.chat(
                model=self.llm_model,
                messages=messages,
                stream=is_stream,
            )

            if not is_stream:
                generation_time = time.time() - generation_start
                logger.debug(f"Response generated in {generation_time:.2f} seconds")

            return response if is_stream else (response["message"]["content"])
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if is_stream:
                return iter([{"message": {"content": error_msg}}])
            else:
                return error_msg

    def detect_query_type(self, query: str, relevance_threshold: float = 0.03) -> str:
        """Use LLM to classify the query type into cybersecurity-related, compliment, complaint, chitchat, or off-topic."""
        logger.debug(f"Detecting query type for: '{query}'")

        system_prompt = """
        You are an expert at classifying cybersecurity and information security queries. Your task is to categorize each query into EXACTLY ONE of these categories:

        1. "cybersecurity-related" - Questions about cybersecurity standards, frameworks, controls, policies, risk management, compliance, threats, vulnerabilities, security functions, governance, etc.
        Examples:
        - "What are the six functions of NIST CSF 2.0?"
        - "What does the Govern function cover?"
        - "How do I implement access control?"
        - "What is the Protect function?"
        - "What is supply chain risk management?"
        - "How does the Detect function work?"
        - "What are CSF Tiers?"
        - "What is a CSF Profile?"
        - "How do I recover from a cybersecurity incident?"
        - "What subcategories does Identify cover?"

        2. "compliment" - Positive feedback or appreciation
        Examples:
        - "Thank you for the information"
        - "That's very helpful"
        - "Great explanation"
        - "You're doing a great job"
        - "You're providing relevant information"

        3. "complaint" - Negative feedback or dissatisfaction
        Examples:
        - "That's not what I asked"
        - "I'm not happy with your answer"
        - "Your answer is wrong"
        - "You are responding slow"
        - "That's not the answer I expected"

        4. "chitchat" - General conversation not directly related to cybersecurity
        Examples:
        - "How are you today?"
        - "What's your name?"
        - "Tell me a joke"
        - "How was your day?"
        - "Hey, nice to meet you!"

        5. "off-topic" - Questions completely unrelated to cybersecurity or information security
        Examples:
        - "How do I fix my car?"
        - "What's the capital of France?"
        - "Can you write me a poem?"
        - "What is the hotel check-in time?"
        - "Tell me about cooking recipes"

        IMPORTANT RULES:
        - Follow-up questions about cybersecurity topics should be classified as "cybersecurity-related" even if they are brief
        - Questions starting with "Is it...", "Does it...", "How much...", "What is..." often refer to previously mentioned cybersecurity topics
        - Treat ambiguous queries that could reasonably be about cybersecurity as "cybersecurity-related"
        - WHEN IN DOUBT, classify as "cybersecurity-related"

        Return ONLY the category name, with no explanation.
        """

        user_prompt = f'Classify this query: "{query}"'

        try:
            initial_classification = (
                self.generate_answer(system_prompt, user_prompt, is_stream=False)
                .lower()
                .strip()
            )

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

            logger.debug(f"Initial classification: '{classification}'")

            if classification == "cybersecurity-related":
                return classification

            if classification in ["chitchat", "off-topic"]:
                relevant_docs, top_similarity = self.search(query, top_k=1)

                logger.debug(
                    f"Query: '{query}' | Initial classification: {classification} | Top similarity: {top_similarity:.4f}"
                )

                if top_similarity >= relevance_threshold:
                    logger.debug(f"Reclassifying to cybersecurity-related due to high document similarity: {top_similarity:.4f}")
                    return "cybersecurity-related"

            return classification

        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            return "cybersecurity-related"

    def contextualize_query(self, current_query: str) -> str:
        """Enhance the current query with context from conversation history."""
        if not self.conversation_history:
            return current_query

        logger.debug(f"Contextualizing query: '{current_query}'")

        if len(current_query.split()) > 10:
            logger.debug("Query is detailed enough, no contextualization needed")
            return current_query

        history_context = (
            "The conversation history is ordered from oldest to most recent:\n\n"
        )
        for idx, (q, a) in enumerate(self.conversation_history[-5:]):
            history_context += f"Exchange {idx + 1}:\nUser: {q}\nAssistant: {a}\n\n"

        system_prompt = """You are a query reformulation system for a cybersecurity standards assistant. Your job is to take a user's query and reformulate it to be self-sufficient by incorporating relevant context from the conversation history, while preserving the original intent.

RULES:
1. Create a single, concise sentence that captures both the current query and relevant context
2. Never add your own opinions, reasoning, explanations, or commentary
3. Preserve all details from the user's current query
4. Be sensitive to topic changes - when a topic changes, do not carry over unrelated context
5. For ambiguous references, prioritize the most relevant (not necessarily the most recent) conversation
6. Never answer the question - only reformulate it
7. If the query is already self-sufficient, make minimal changes or return it as is
8. Use natural language as if the user had included all relevant context themselves
9. If the user's intent is a question, always ensure the reformulated text is also a question

EXAMPLES:

Example 1 - Basic context addition:
User: "What does the Protect function cover?"
Reformulated query: "What areas and subcategories does the NIST CSF 2.0 Protect (PR) function cover?"

Example 2 - Topic change recognition:
User: "What does the Protect function cover?"
Assistant: [Reply on Protect function...]
User: "What about Detect?"
Reformulated query: "What does the NIST CSF 2.0 Detect (DE) function cover?"
DO NOT: "What does the Protect function and Detect function cover?" [INCORRECT - forces old topic into new question]

Example 3 - Pronoun resolution:
User: "What is the Govern function?"
Assistant: [Reply on Govern function...]
User: "Is it new in CSF 2.0?"
Reformulated query: "Is the Govern function new in NIST CSF 2.0?"

Example 4 - Preserve question format:
User: "Tell me about CSF Tiers."
Assistant: [Reply on CSF Tiers...]
User: "How many are there?"
Reformulated query: "How many Tiers are defined in NIST CSF 2.0?"

DO NOT add commentary or explanations.
DO NOT exceed one sentence unless absolutely necessary.
Always maintain the question format when the user's intent is a question."""

        user_prompt = f"""
Conversation history:
{history_context}

Current query: "{current_query}"

Reformulated query:
        """
        try:
            reformulated_query = self.generate_answer(
                system_prompt, user_prompt, is_stream=False
            ).strip()
            logger.debug(f"Reformulated query: '{reformulated_query}'")

            if ":" in reformulated_query:
                parts = reformulated_query.split(":", 1)
                if len(parts) > 1:
                    reformulated_query = parts[1].strip()

            reformulated_query = reformulated_query.strip("\"'")

            problematic_phrases = [
                "based on",
                "according to",
                "from our conversation",
                "as mentioned",
                "earlier you asked",
                "you asked about",
            ]

            if (
                len(reformulated_query.split()) > 25
                or any(
                    phrase in reformulated_query.lower()
                    for phrase in problematic_phrases
                )
                or ("." in reformulated_query and "?" not in reformulated_query)
            ):
                logger.debug("Reformulation looks problematic, falling back to original query")
                return current_query

            if reformulated_query.lower() == current_query.lower():
                logger.debug("Reformulation is identical to original query")
                return current_query

            return reformulated_query
        except Exception as e:
            logger.error(f"Error in query contextualization: {str(e)}")
            return current_query

    def handle_non_cybersecurity_query(self, query_type: str, query: str) -> Union[Iterator[Dict], str]:
        """Handle compliments, complaints, chitchat, and off-topic queries using LLM."""
        logger.debug(f"Handling {query_type} query: '{query}'")

        if query_type == "compliment":
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has given you a compliment or thanked you.

            Respond with:
            1. A brief, gracious acknowledgment of their thanks
            2. An offer to help with any cybersecurity standards or framework questions they might have

            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'User compliment: "{query}"'

        elif query_type == "complaint":
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has expressed dissatisfaction or complained.

            Respond with:
            1. A brief, professional apology for any confusion
            2. An offer to help with specific cybersecurity standards questions

            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'User complaint: "{query}"'

        elif query_type == "chitchat":
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has engaged in general conversation not related to cybersecurity.

            Respond with:
            1. A brief, friendly response
            2. A polite redirection to cybersecurity-related topics

            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'User message: "{query}"'

        else:  # off-topic
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has asked about a topic completely unrelated to cybersecurity.

            Respond with:
            1. A polite explanation that you are designed to assist with cybersecurity standards and frameworks such as NIST CSF 2.0
            2. An offer to help with cybersecurity-related questions

            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'Off-topic question: "{query}"'

        return self.generate_answer(system_prompt, user_prompt)

    def rag(self, query: str) -> Union[Iterator[Dict], str]:
        """Main function to perform conversational document search and answer generation."""
        try:
            logger.debug(f"Processing query: '{query}'")

            query_type = self.detect_query_type(query)
            logger.debug(f"Query type determined: '{query_type}'")

            if query_type != "cybersecurity-related":
                logger.debug(f"Handling non-cybersecurity query of type: {query_type}")
                response_stream = self.handle_non_cybersecurity_query(query_type, query)

                if not isinstance(response_stream, Iterator):
                    self.conversation_history.append((query, response_stream))
                    return iter([{"message": {"content": response_stream}}])

                return response_stream

            contextualized_query = self.contextualize_query(query)
            logger.debug(f"Contextualized query: '{contextualized_query}'")

            context, relevance_score = self.search(contextualized_query)
            logger.debug(f"Top relevance score: {relevance_score:.4f}")

            if self.advanced_mode and self.search_mode == "vector":
                RELEVANCE_THRESHOLD = 0.3
            else:
                RELEVANCE_THRESHOLD = 0.02

            if relevance_score < RELEVANCE_THRESHOLD and context != "No documents available.":
                logger.debug(f"Low relevance score ({relevance_score:.4f}), using fallback response")
                system_prompt = """
                You are an AI assistant specializing in cybersecurity standards and frameworks. You are designed to answer questions about frameworks such as NIST CSF 2.0, ISO 27001, and other cybersecurity standards.

                The user has asked a question that does not match the available document content. Respond with:

                1. A polite explanation that you do not have enough information in your current documents to answer their specific question
                2. An offer to help with other cybersecurity standards questions
                3. A suggestion to check the original standard documents for more specific information

                Keep your response brief and professional.
                """

                user_prompt = f'User question without matching context: "{query}"'

                return self.generate_answer(system_prompt, user_prompt)

            logger.debug("Generating final response with retrieved context")
            system_prompt = """You are an expert AI assistant specializing in cybersecurity standards and frameworks. You have access to a collection of detailed information from cybersecurity standards documents such as the NIST Cybersecurity Framework (CSF) 2.0 and other official cybersecurity standards.

                Your task is to help users by providing accurate, clear, and well-structured answers about cybersecurity standards, controls, functions, categories, and best practices.

                The information you have includes:

                Cybersecurity Framework Functions: Govern (GV), Identify (ID), Protect (PR), Detect (DE), Respond (RS), Recover (RC) — their purposes, categories, and subcategories.
                Risk Management: Organizational risk strategy, risk assessment, risk tolerance, and risk treatment.
                Governance: Cybersecurity policies, roles, responsibilities, supply chain risk management, and oversight.
                Identity and Access Control: Asset management, user authentication, privilege management, and access policies.
                Protective Controls: Data security, platform security, awareness training, and technology infrastructure resilience.
                Detection and Analysis: Continuous monitoring, anomaly detection, and adverse event analysis.
                Incident Response and Recovery: Incident management, communication, mitigation, and recovery planning.
                CSF Profiles and Tiers: Framework customization and maturity assessment.

                If a user's question cannot be answered based on the provided context, respond with:

                "I cannot answer this based on the available documents. Please refer to the official standard documentation or consult your cybersecurity administrator for more information."

                Respond in a clear, structured, and professional manner. Use bullet points or numbered lists when explaining complex topics. Always base your answers on the provided document context.
                """

            user_prompt = f'Based on the following context from cybersecurity standards documents, please answer the question.\nIf the answer cannot be derived from the context, say "I cannot answer this based on the available documents." \n\nContext:\n{context}\n\nQuestion: {contextualized_query}\n\nAnswer:\n'

            logger.debug(f"Retrieved context: {context[:500]}...")

            return self.generate_answer(system_prompt, user_prompt, inject_history=True)

        except Exception as e:
            error_msg = "I apologize, but I encountered an error while processing your question. Please try asking again or consult your cybersecurity administrator for assistance."
            logger.error(f"Error in RAG process: {str(e)}")
            return iter([{"message": {"content": error_msg}}])


def chunk_text_with_overlap(text: str, chunk_size: int = 450, overlap_size: int = 100) -> List[str]:
    """Split text into chunks with overlap. Optimized for dense cybersecurity standards documents."""
    if not text:
        return ["No text provided"]

    words = text.split()

    words_per_chunk = chunk_size
    overlap_words = overlap_size

    if not words:
        return ["No text provided"]

    if len(words) <= words_per_chunk:
        return [text]

    logger.debug(f"Chunking text with {len(words)} words into chunks of ~{words_per_chunk} words with {overlap_words} word overlap")

    chunks = []
    i = 0
    while i < len(words):
        chunk_end = min(i + words_per_chunk, len(words))
        chunk = " ".join(words[i:chunk_end])
        chunks.append(chunk)

        i += words_per_chunk - overlap_words

        if i + words_per_chunk > len(words) and i < len(words):
            remaining = len(words) - i
            if remaining > overlap_words:
                chunks.append(" ".join(words[i:]))
            break

    improved_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > (words_per_chunk / 2):
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

    logger.debug(f"Created {len(chunks)} initial chunks, {len(improved_chunks)} improved chunks")

    return improved_chunks if improved_chunks else chunks
