# MedGemma RAG Performance Evaluation

This project evaluates the performance of the MedGemma LLM in generating hospital discharge summaries using a Retrieval-Augmented Generation (RAG) approach. It combines real-time patient transcripts with OpenEMR data and enforces a strict output schema.

## System Architecture

The core logic is distributed across two main files: `prompt.py` and `main.py`.

### 1. `prompt.py`: Prompt Engineering & LLM Setup

This file is responsible for constructing the input for the Language Model and managing the connection to it.

*   **Prompt Template (`get_discharge_summary_prompt`)**:
    *   Defines a strict **SOAP (Subjective, Objective, Assessment, Plan)** structure for the discharge summary.
    *   **Input Sources**:
        *   **Transcript**: Used for subjective data (HPI, ROS, current plan).
        *   **OpenEMR Extract**: Used for objective data (Vitals, PMH, Meds, Labs).
        *   **Schema Guide**: A JSON structure that dictates the exact sections and subsections required.
    *   **Enforcement**: The prompt includes strict instructions to:
        *   Prioritize the transcript for current encounter details.
        *   Filter medications to include only active/ongoing ones.
        *   Act as a "Form Filler" for Labs & Imaging, strictly following the provided schema.
        *   Output pure narrative text without leaking JSON or schema keys.

*   **LLM Connection (`get_llm`)**:
    *   Initializes the connection to the **MedGemma-27b-it** model.
    *   Uses a **Cloudflare tunnel** URL to access the model hosted remotely.
    *   Configures generation parameters like `temperature=0.3` to balance creativity with adherence to facts.

### 2. `main.py`: RAG Pipeline & Evaluation Orchestration

This file serves as the main entry point and orchestrator of the entire workflow.

*   **RAG Pipeline**:
    *   **Indexing**:
        *   **Source**: MIMIC Clinical Notes.
        *   **Preprocessing**: Shared with a local instance of **MedGemma-27b** to extract the note structure while removing all Patient Health Information (PHI).
        *   **Storage**: The extracted structures were saved to `data/all_notes_structure.json`, with metadata including the associated Disease.
        *   **Chunking**: Each chunk corresponds to one unique MIMIC Note Structure.
        *   **Embedding**: Created using **SentenceBERT (`all-MiniLM-L6-v2`)** and stored in **ChromaDB** for efficient retrieval.
    *   **Retrieval (`retrieve`)**:
        *   **Disease Extraction**: Uses **MedGemma-27b-it** to analyze the transcript and extract the primary chronic condition or reason for admission.
        *   **Semantic Search**: Uses **SentenceBERT (`all-MiniLM-L6-v2`)** to embed the extracted disease name and compare it against the metadata in the ChromaDB vector store.
        *   **Top-k Retrieval**: Retrieves the **top 2** most semantically similar "Schema Guides" (note structures) based on cosine similarity.
        *   **Context Construction**: Combines the content of these top 2 chunks to provide a richer context for the generation phase.
    *   **Generation (`generate`)**:
        *   **Prompt Construction**: Dynamically assembles a prompt using:
            *   **Transcript**: For Subjective data (HPI, ROS).
            *   **OpenEMR**: For Objective data (Vitals, Meds, Labs).
            *   **Retrieved Schema**: As a structural skeleton (not content).
        *   **Model**: **MedGemma-27b-it** (via Cloudflare tunnel).
        *   **Enforcement**: The model is instructed to strictly follow the retrieved schema's structure, filling in "Not available" for missing data rather than hallucinating or summarizing.

*   **Evaluation (`Phase 3`)**:
    *   Compares the generated summary against a **Reference Summary** (Human-written Gold Standard).
    *   **Metrics & Relevance**:
        *   **SBERT (Sentence-BERT)**:
            *   **What**: Uses siamese networks to create embeddings for entire sentences/paragraphs, comparing them via cosine similarity.
            *   **Relevance**: Measures **Global Semantic Coherence**. It checks if the *overall narrative and meaning* of the generated summary align with the reference, regardless of exact phrasing.
        *   **BERTScore**:
            *   **What**: Computes similarity using contextual embeddings at the token level, matching each generated token to the most similar reference token.
            *   **Relevance**: Measures **Local Semantic Accuracy**. It ensures that specific clinical details (medications, dosages, symptoms) are semantically correct, even if synonyms are used. It is far more robust than n-gram matching (BLEU).
        *   **BLEU & ROUGE-L**: Traditional metrics used for measuring exact lexical overlap and structural recall.
    *   **Results Interpretation**:
        *   **Score Range**: Across 6 different patients, the system consistently achieved SBERT and BERTScore values between **0.80 - 0.90**.
        *   **Implication**: This high and stable range indicates **Clinical Equivalence**. The RAG pipeline reliably captures the core clinical facts and narrative structure of the human reference, proving that the model is not just "guessing" but is grounded in the source data.
    *   **Output**: Saves the generated summary and all metrics to `results/evaluation.txt`.

## Workflow Summary

1.  **Input**: The system reads a patient transcript (`data/transcription_toma.txt`) and an OpenEMR extract.
2.  **Disease Extraction**: `main.py` asks the LLM to identify the primary condition from the transcript.
3.  **Schema Retrieval**: The system searches the vector database for the standard note structure for that condition.
4.  **Prompt Construction**: `prompt.py` formats the data and schema into a strict instruction set.
5.  **Generation**: MedGemma generates the discharge summary.
6.  **Evaluation**: The generated text is compared to a gold-standard reference, and metrics are calculated.

<img width="2784" height="1536" alt="Diagram" src="https://github.com/user-attachments/assets/c8244b57-adb2-4e66-a553-8b522c2b1b8f" />

