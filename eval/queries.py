"""
queries.py
----------
Stage 1: Test queries + ground truth answers for evaluating the RAG system.

Edit/extend this list freely. Keep ground_truth answers short, factual, and
written the way a human grader would phrase the "correct" answer -- the
semantic-similarity judges compare the MODEL RESPONSE against this text,
so it should capture the key fact(s), not be a full essay.

If your docs/ folder contains the full NIST.CSWP.29.pdf (not just the
built-in summary), feel free to add more specific questions that only the
full PDF would answer correctly.
"""

TEST_QUERIES = [
    {
        "id": "q01",
        "query": "What are the six core functions of the NIST Cybersecurity Framework 2.0?",
        "ground_truth": "The six core functions are Govern, Identify, Protect, Detect, Respond, and Recover.",
    },
    {
        "id": "q02",
        "query": "What is new about the Govern function in CSF 2.0 compared to CSF 1.1?",
        "ground_truth": "Govern is a new function added in CSF 2.0 that establishes and monitors the organization's cybersecurity risk management strategy, expectations, and policy, reflecting governance importance at the executive and board level.",
    },
    {
        "id": "q03",
        "query": "What does the Identify function help an organization do?",
        "ground_truth": "The Identify function helps the organization understand its current cybersecurity risks to systems, people, assets, data, and capabilities.",
    },
    {
        "id": "q04",
        "query": "What does the Protect function cover?",
        "ground_truth": "The Protect function uses safeguards to prevent or reduce cybersecurity risks, including access control, data security, and awareness training.",
    },
    {
        "id": "q05",
        "query": "What is the purpose of the Detect function?",
        "ground_truth": "The Detect function finds and analyzes possible cybersecurity attacks and compromises through continuous monitoring.",
    },
    {
        "id": "q06",
        "query": "What does the Respond function involve?",
        "ground_truth": "The Respond function takes action regarding a detected cybersecurity incident, including incident management and communication.",
    },
    {
        "id": "q07",
        "query": "What is the goal of the Recover function?",
        "ground_truth": "The Recover function restores assets and operations impacted by a cybersecurity incident, including recovery planning and improvements.",
    },
    {
        "id": "q08",
        "query": "What is covered under Risk Management Strategy, GV.RM?",
        "ground_truth": "GV.RM ensures the organization's priorities, constraints, risk tolerance, and assumptions are established, communicated, and used to support cybersecurity risk decisions.",
    },
    {
        "id": "q09",
        "query": "What does Identity Management and Access Control, PR.AA, ensure?",
        "ground_truth": "PR.AA ensures that access to physical and logical assets is limited to authorized users, processes, and devices.",
    },
    {
        "id": "q10",
        "query": "What is Data Security, PR.DS, responsible for?",
        "ground_truth": "PR.DS ensures data-at-rest, in-transit, and in-use are managed according to the organization's risk strategy to protect confidentiality, integrity, and availability.",
    },
    {
        "id": "q11",
        "query": "What is a CSF Profile?",
        "ground_truth": "A CSF Profile represents the alignment of the CSF Core to an organization's requirements, risk tolerance, and resources to support prioritization of cybersecurity activities.",
    },
    {
        "id": "q12",
        "query": "What do CSF Tiers describe, and what is the range?",
        "ground_truth": "CSF Tiers describe the maturity of an organization's cybersecurity risk governance and management practices, ranging from Tier 1 (Partial) to Tier 4 (Adaptive).",
    },
    {
        "id": "q13",
        "query": "How is supply chain risk management addressed in CSF 2.0?",
        "ground_truth": "Supply chain risk management is addressed under the Govern function to ensure cybersecurity risks in the supply chain are identified, assessed, and managed.",
    },
    {
        "id": "q14",
        "query": "What is Asset Management, ID.AM, about?",
        "ground_truth": "ID.AM involves identifying and managing assets such as data, hardware, software, systems, and facilities that enable the organization to achieve business purposes.",
    },
    {
        "id": "q15",
        "query": "What does the CSF Core consist of?",
        "ground_truth": "The CSF Core provides a set of cybersecurity outcomes organized as Functions, Categories, and Subcategories that can be customized for any organization.",
    },
]
