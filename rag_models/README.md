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
        *   **Embedding**: The on-disk Chroma index in `vectorDB/chroma_schema_improved/` was built with BioBERT (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) for legacy reasons but those vectors are **not** used at retrieval time — see below. The retriever re-encodes disease metadata strings with `all-MiniLM-L6-v2` at runtime.
    *   **Retrieval (`retrieve`)**:
        *   **Disease Extraction**: Uses the same LLM that will generate the summary to analyze the transcript and extract the primary clinical condition (max 20 tokens).
        *   **Semantic Search (live)**: Loads disease metadata strings from Chroma via `vector_store.get(...)` (Chroma's BioBERT vectors are bypassed) and re-encodes them with `all-MiniLM-L6-v2` (`SentenceTransformer`). The detected disease is encoded with the same model, and cosine similarity selects the top-k schemas. **Both query and candidates therefore live in the same MiniLM embedding space**, which is why the embedding-substitution ablation swaps the `SentenceTransformer` model rather than rebuilding the Chroma index.
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

## RAG Ablation Study (Institutional 6-Case Evaluation)

Three ablation variants were run on the same six institutional conversations used for the reference-based evaluation, holding all other pipeline components constant. Each configuration was evaluated over **3 independent runs** per LLM.

### Design

1. **No-RAG baseline.** Schema retrieval disabled; the LLM prompt contained only the conversation transcript. Implemented by replacing the `retrieve()` call in `pipeline/run_fareez_summaries.py` with an empty schema context (`retrieved_schemas = ''`) and writing to a separate output directory.
2. **Retrieval-depth sweep.** Top-k schemas varied over k ∈ {1, 2, 3, 5}, implemented by varying the `n_results` argument to `collection.query(...)` in the same file.
3. **Embedding-model substitution.** `all-MiniLM-L6-v2` replaced with ClinicalBERT and PubMedBERT. Requires rebuilding the ChromaDB index with `pipeline/create_vector_db.py` pointing at the alternate embedding.

A **FHIR-context ablation is not applicable** to this architecture: the summarization prompt contains only the conversation transcript and retrieved SOAP schemas as plain text. FHIR-derived structured resources (Patient, Condition, MedicationRequest, AllergyIntolerance, Procedure, Observation) are used exclusively for clinician display and for note write-back — never as input to the generator — so there is no FHIR input variable to remove.

### Metrics

Six automated metrics were computed per run (see `evaluation/`):

| Metric | What it measures | Dimension |
|-------|------------------|-----------|
| BLEU | n-gram overlap | Lexical |
| ROUGE-L | longest common subsequence | Structural |
| SBERT coherence | sentence-embedding cosine similarity | Global semantic |
| BERTScore F1 | token-level contextual-embedding F1 | Local semantic |
| scispaCy entity recall | UMLS CUI recall via scispaCy | Clinical entity |
| MedCAT entity recall | UMLS CUI recall via MedCAT | Clinical entity |

### Results

Canonical numeric results are stored in `results/institutional/`:

- **`table2_rag_6case.csv`** — RAG on (k=2, all-MiniLM-L6-v2). Corresponds to Table 2 in the paper.
- **`table2b_norag_6case.csv`** — no-RAG baseline. Corresponds to Table 2b in the paper.
- **`ablation_analysis.ipynb`** — analysis notebook that loads both CSVs, computes the RAG-vs-no-RAG deltas, and verifies every quantitative claim made in the paper's ablation paragraphs (MedCAT drop range 0.020–0.060, ROUGE-L drop range 0.005–0.050 with LLaMA-3.1-8B showing the largest drop, BERTScore F1 drop range 0.020–0.060, SBERT drops for Qwen3-32B and MedGemma-4B, and near-constant inference time).

### Key findings

- **Disabling retrieval degrades every metric for every model.** MedCAT recall dropped by 0.020–0.060, ROUGE-L by 0.005–0.050, BERTScore F1 by 0.020–0.060. The largest SBERT coherence drops were observed for Qwen3-32B (0.507 → 0.416) and MedGemma-4B (0.773 → 0.693), indicating that smaller or less instruction-tuned models benefit disproportionately from schema scaffolding.
- **k=3 and k=5 substantially degraded quality** vs. k=2. Higher k pulled in loosely related schemas, lowering mean retrieval similarity and causing the LLM to blend organ-system-inappropriate template language. Production setting: **k=2**.
- **ClinicalBERT and PubMedBERT matched all-MiniLM-L6-v2 on retrieval quality** but added 0.92–1.32 s retrieval latency per call. Production embedding: **all-MiniLM-L6-v2**.
- **Inference time is unchanged** when retrieval is disabled (max |Δt| ≈ 2 s across models), confirming that LLM generation, not retrieval, dominates latency.

### Reproducing end-to-end

Full end-to-end reproduction requires the six institutional transcripts, paired OpenEMR extracts, and reference SOAP notes, which contain institution-specific identifiers from the IU Indianapolis simulation set and are **not** included in this repository. The canonical numeric outputs in `results/institutional/` are therefore the authoritative artifact for the ablation analysis. The end-to-end scripts (`models/*_modal.py`, `pipeline/run_fareez_summaries.py`, `pipeline/create_vector_db.py`) reproduce Fareez OSCE results, which use the same pipeline.

