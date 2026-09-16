import heapq
import logging
import numpy as np
import ollama
import re


def extract_identifier_terms(text: str):
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


def embed_text(text: str, embedding_model="nomic-embed-text"):
    """Embed text using Ollama's specified embedding model (default: nomic-embed-text)."""
    response = ollama.embeddings(model=embedding_model, prompt=text)
    embedding = response["embedding"]
    return np.array(embedding)


def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


def topk(arr, k):
    """Get the k largest elements and their indexes."""
    return heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])


def search(query, documents, document_embeddings, top_k=5):
    """Search for the most relevant document based on the cosine similarity of embeddings."""
    query_embedding = embed_text(query)

    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
    boosted_scores = [min(1.0, similarities[i] + acronym_boost(query, documents[i])) for i in range(len(documents))]

    topk_indices = topk(boosted_scores, top_k)
    results = []
    for i in topk_indices:
        results.append(f"Document: {documents[i]}\nSimilarity Score: {boosted_scores[i]:.4f}")

    result = '\n\n'.join(results)
    return result


def handle_user_input():
    """Function to handle user input."""
    if prompt := input("Enter your cybersecurity standards query: "):
        return prompt


def main():
    """Main function to perform semantic search in a loop."""
    while True:
        print("\n" + "="*60)
        print("Cybersecurity Standards — Semantic Search")
        print("="*60)

        user_input = handle_user_input()
        if user_input:
            relevant_documents = search(user_input, documents, document_embeddings)
            print(f"\nRelevant Documents for your query: '{user_input}'\n")
            print("-"*60)
            print(f"{relevant_documents}")
            print("-"*60)

        continue_query = input("\nDo you want to search again? (y/Y to continue, any other key to quit): ")
        if continue_query.lower() != "y":
            print("\nExiting!")
            break


if __name__ == "__main__":
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

    document_embeddings = [embed_text(doc) for doc in documents]
    main()
