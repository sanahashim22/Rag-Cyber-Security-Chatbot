import heapq
import numpy as np
import ollama
import re
from typing import List, Dict, Tuple, Any


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


class ConvoRAG:
    def __init__(self, documents: List[str], embedding_model: str = "nomic-embed-text", llm_model: str = "llama3.2"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.document_embeddings = [self.embed_text(doc) for doc in documents]
        self.conversation_history = []

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Ollama's specified embedding model."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        embedding = response["embedding"]
        return np.array(embedding)

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def topk(self, arr: List[float], k: int) -> List[int]:
        """Get the k largest elements and their indexes."""
        topk_indices = heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
        return topk_indices

    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Search for the most relevant documents based on cosine similarity of embeddings."""
        query_embedding = self.embed_text(query)

        similarities = []
        boosted_scores = []
        for idx, doc_embedding in enumerate(self.document_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
            boosted_scores.append(min(1.0, similarity + acronym_boost(query, self.documents[idx])))

        topk_indices = self.topk(boosted_scores, top_k)

        result = '\n'.join([self.documents[i] for i in topk_indices])
        return result, boosted_scores[topk_indices[0]]

    def generate_answer(self, system_prompt: str, user_prompt: str, inject_history: bool = False) -> str:
        """Generate a response using the provided system and user prompts."""
        messages = [{'role': 'system', 'content': system_prompt}]
        if inject_history and self.conversation_history:
            for (q, a) in self.conversation_history[-3:]:  # last 3 turns only
                messages.append({'role': 'user', 'content': q})
                messages.append({'role': 'assistant', 'content': a})
        messages.append({'role': 'user', 'content': user_prompt})

        response = ollama.chat(
            model=self.llm_model,
            messages=messages
        )
        return response['message']['content']

    def detect_query_type(self, query: str) -> str:
        """Use LLM to classify the query type into cybersecurity-related, compliment, complaint, chitchat, or off-topic."""
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
        
        IMPORTANT:
        - Follow-up questions about cybersecurity topics should be classified as "cybersecurity-related" even if they are brief
        - Questions starting with "Is it..." or "Does it..." or "What about..." often refer to previously mentioned cybersecurity topics
        - Treat ambiguous queries that could reasonably be about cybersecurity as "cybersecurity-related"
        - WHEN IN DOUBT, classify as "cybersecurity-related"
        
        Return ONLY the category name, with no explanation.
        """

        user_prompt = f'Classify this query: "{query}"'

        response = self.generate_answer(system_prompt, user_prompt).lower().strip()

        # Check for exact category name first, then fallback keywords
        if "cybersecurity-related" in response:
            return "cybersecurity-related"
        elif "off-topic" in response or "offtopic" in response:
            return "off-topic"
        elif "compliment" in response:
            return "compliment"
        elif "complaint" in response:
            return "complaint"
        elif "chitchat" in response:
            return "chitchat"
        elif "cybersecurity" in response:
            return "cybersecurity-related"
        else:
            return "cybersecurity-related"  # safe default

    def contextualize_query(self, current_query: str) -> str:
        """Enhance the current query with context from conversation history."""
        if not self.conversation_history:
            return current_query

        if len(current_query.split()) > 10:
            return current_query

        history_context = "The conversation history is ordered from oldest to most recent:\n\n"
        for idx, (q, a) in enumerate(self.conversation_history[-5:]):
            history_context += f"Exchange {idx+1}:\nUser: {q}\nAssistant: {a}\n\n"

        system_prompt = """
        You are a query reformulation system for a cybersecurity standards assistant.
        
        Your task is to analyze the current query in the context of the conversation history and determine:
        1. IF the current query is a follow-up question that needs context from previous exchanges
        2. If it is, WHICH previous cybersecurity topic it most likely refers to
        3. HOW to reformulate the query to be completely self-contained
        
        IMPORTANT RULES:
        - The conversation history is ordered from oldest to most recent
        - Brief queries like "What does it cover?" or "Is it required?" are almost always follow-ups
        - Use ONLY information explicitly mentioned in the conversation history
        - NEVER include information not present in the conversation
        - NEVER answer the query — just reformulate it
        - NEVER provide explanations outside the reformulated query
        - Keep reformulations concise (under 20 words when possible)
        - Reply with ONLY the reformulated query or the original query if no reformulation is needed
        
        EXAMPLES:
        1. History: User asked about the Protect function, then the Detect function. Current query: "What subcategories does it have?"
           Response: "What subcategories does the NIST CSF 2.0 Detect function have?"
        
        2. History: User asked about CSF Tiers, then CSF Profiles. Current query: "Is it mandatory?"
           Response: "Is a CSF Profile mandatory for organizations?"
        """

        user_prompt = f"""
        Conversation history:
        {history_context}
        
        Current query: "{current_query}"
        
        Reformulated query (if needed):
        """

        reformulated_query = self.generate_answer(system_prompt, user_prompt).strip()

        if ":" in reformulated_query:
            reformulated_query = reformulated_query.split(":", 1)[1].strip()

        problematic_phrases = [
            "based on", "according to", "from our conversation",
            "as mentioned", "earlier you asked", "you asked about"
        ]

        if (len(reformulated_query.split()) > 25 or
                any(phrase in reformulated_query.lower() for phrase in problematic_phrases) or
                "." in reformulated_query and "?" not in reformulated_query):
            return current_query

        if reformulated_query.lower() == current_query.lower():
            return current_query

        return reformulated_query

    def handle_non_cybersecurity_query(self, query_type: str, query: str) -> str:
        """Handle compliments, complaints, chitchat, and off-topic queries using LLM."""
        system_prompt = """
        You are an AI assistant for a Cybersecurity Standards information system. Respond to the user message appropriately based on its category.
        
        Keep your response:
        1. Professional and courteous
        2. Relatively brief (2-3 sentences)
        3. Empathetic when needed
        4. Gently redirecting to cybersecurity standards topics when appropriate
        
        Do not include the category in your response.
        You are only able to help with cybersecurity standards and frameworks such as NIST CSF 2.0.
        """

        user_prompt = f"""
        Message category: {query_type}
        User message: "{query}"
        
        Please respond appropriately to this message.
        """

        return self.generate_answer(system_prompt, user_prompt)

    def rag(self, query: str) -> str:
        """Main function to perform conversational document search and answer generation."""
        query_type = self.detect_query_type(query)

        if query_type != "cybersecurity-related":
            response = self.handle_non_cybersecurity_query(query_type, query)
            self.conversation_history.append((query, response))
            return response

        contextualized_query = self.contextualize_query(query)

        context, _ = self.search(contextualized_query)

        system_prompt = '''You are an expert AI assistant specializing in cybersecurity standards and frameworks. You have access to detailed information from cybersecurity standards documents such as the NIST Cybersecurity Framework (CSF) 2.0.

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

        Respond in a clear, structured, and professional manner. Use bullet points or numbered lists when explaining complex topics.
        '''

        user_prompt = f"Based on the following context from cybersecurity standards documents, please answer the question.\nIf the answer cannot be derived from the context, say \"I cannot answer this based on the available documents.\" \n\nContext:\n{context}\n\nQuestion: {contextualized_query}\n\nAnswer:\n"

        answer = self.generate_answer(system_prompt, user_prompt, inject_history=True)

        self.conversation_history.append((query, answer))

        return answer


def main():
    """
    Main function to create and run the conversational cybersecurity RAG system.
    """
    # NIST CSF 2.0 — Key concepts used as the knowledge base for this module.
    documents = [
        # CSF Overview
        "The NIST Cybersecurity Framework (CSF) 2.0 provides guidance to manage cybersecurity risks and is organized around six core Functions: Govern, Identify, Protect, Detect, Respond, and Recover.",
        "NIST CSF 2.0 introduced the Govern (GV) function as a new addition over CSF 1.1, reflecting the importance of cybersecurity governance at the executive and board level.",
        "CSF Profiles represent the alignment of the CSF Core to an organization's requirements, risk tolerance, and resources to support prioritization of cybersecurity activities.",
        "CSF Tiers describe the maturity of an organization's cybersecurity risk governance ranging from Tier 1 (Partial) to Tier 4 (Adaptive).",
        # Govern Function
        "The Govern (GV) Function establishes and monitors the organization's cybersecurity risk management strategy, expectations, and policy.",
        "Organizational Context (GV.OC) ensures that the organizational mission is understood and that cybersecurity risk management supports the organization's objectives.",
        "Risk Management Strategy (GV.RM) ensures that priorities, constraints, risk tolerance, and assumptions are established, communicated, and used to support cybersecurity risk decisions.",
        "Supply chain risk management under Govern ensures cybersecurity risks in the supply chain are identified, assessed, and managed.",
        # Identify Function
        "The Identify (ID) Function helps the organization understand its current cybersecurity risks to systems, people, assets, data, and capabilities.",
        "Asset Management (ID.AM) involves identifying and managing assets including data, hardware, software, systems, and facilities that enable the organization to achieve its purposes.",
        "Risk Assessment (ID.RA) identifies and prioritizes risks to the organization's operations, assets, and individuals based on threat and vulnerability information.",
        # Protect Function
        "The Protect (PR) Function uses safeguards to prevent or reduce cybersecurity risks including access control, data security, and training.",
        "Identity Management and Access Control (PR.AA) ensures that access to physical and logical assets is limited to authorized users, processes, and devices.",
        "Awareness and Training (PR.AT) ensures that personnel are provided cybersecurity awareness education and trained to perform their cybersecurity-related duties.",
        "Data Security (PR.DS) ensures that data-at-rest, in-transit, and in-use are managed to protect confidentiality, integrity, and availability.",
        "Platform Security (PR.PS) manages hardware, software, and services of physical and virtual platforms according to the organization's risk strategy.",
        "Technology Infrastructure Resilience (PR.IR) ensures that security architectures are managed to protect asset confidentiality, integrity, and availability.",
        # Detect Function
        "The Detect (DE) Function finds and analyzes possible cybersecurity attacks and compromises through continuous monitoring.",
        "Adverse Event Analysis (DE.AE) ensures that anomalies, indicators of compromise, and other potentially adverse events are analyzed to characterize events and detect cybersecurity incidents.",
        # Respond Function
        "The Respond (RS) Function takes action regarding a detected cybersecurity incident including incident management and communication.",
        "Incident Management (RS.MA) covers the response to detected cybersecurity incidents including execution of the response plan to minimize the impact.",
        "Incident Analysis (RS.AN) ensures that investigations are conducted to support effective response, forensics, and recovery activities.",
        "Incident Response Reporting and Communication (RS.CO) covers response activities coordinated with internal and external stakeholders as required by laws, regulations, or policies.",
        "Incident Mitigation (RS.MI) ensures that activities are performed to prevent expansion of an event and mitigate its effects.",
        # Recover Function
        "The Recover (RC) Function restores assets and operations that were impacted by a cybersecurity incident including recovery planning and improvements.",
        "Incident Recovery Plan Execution (RC.RP) ensures that restoration activities are performed to ensure operational availability of systems and services affected by cybersecurity incidents.",
        "Incident Recovery Communication (RC.CO) ensures that restoration activities are coordinated with internal and external parties such as Internet Service Providers, vendors, and other CSIRTs.",
    ]

    rag_system = ConvoRAG(documents)

    print("\n" + "="*60)
    print("Cybersecurity Standards — Conversational RAG System")
    print("Powered by NIST CSF 2.0")
    print("="*60)

    while True:
        print("\n" + "-"*60)
        user_input = input("Ask a question about cybersecurity standards: ")

        if user_input:
            answer = rag_system.rag(user_input)
            print(f"\nAssistant: {answer}")

        print("-"*60)
        continue_query = input("\nDo you want to ask another question? (y/Y to continue, any other key to quit): ")
        if continue_query.lower() != "y":
            print("\nExiting!")
            break


if __name__ == "__main__":
    main()
