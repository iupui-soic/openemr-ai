"""
Shared Evaluator Service - Deploy once, use by all model pipelines.

This service provides evaluation metrics for medical summarization:
- BLEU, ROUGE-L, SBERT Coherence, BERTScore F1
- scispaCy Entity Recall (CUI-based)
- MedCAT Entity Recall (CUI-based)

Usage:
    modal deploy shared_evaluator_service.py      # Deploy before running pipelines
    modal app stop shared-evaluator-service       # Stop after all pipelines complete

Calling from other pipelines:
    evaluator_app = modal.App.lookup("shared-evaluator-service")
    SummaryEvaluator = evaluator_app.cls("SummaryEvaluator")
    evaluator = SummaryEvaluator()
    metrics = evaluator.evaluate.remote(generated_text, reference_text)
"""

import modal
from typing import Dict

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("shared-evaluator-service")

# ============================================================================
# Evaluator Image (scispaCy + MedCAT + existing metrics)
# ============================================================================

evaluator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Pin numpy first to ensure compatibility
        "numpy>=1.26.0,<2.0.0",
        # Existing evaluation dependencies
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.13",
        "sentence-transformers>=2.2.2",
        "pandas>=2.0.0",
        # scispaCy dependencies - pin compatible versions
        "spacy>=3.7.0,<3.8.0",
        "scispacy>=0.5.4",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz",
        # MedCAT dependencies
        "medcat[spacy,dict-ner]>=2.0.0",
    )
    .run_commands(
        # Install wget first
        "apt-get update && apt-get install -y wget unzip",
        # Download MedCAT model
        "mkdir -p /medcat_models",
        "wget -q https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip -O /medcat_models/medmen_wstatus_2021_oct.zip",
    )
)


# ============================================================================
# Shared Summary Evaluator Class
# ============================================================================

@app.cls(
    image=evaluator_image,
    timeout=1800,
    cpu=2,
    memory=8192,
    container_idle_timeout=600,  # Auto-sleep after 10 min idle (cost saving)
)
class SummaryEvaluator:
    """
    Shared evaluator for medical summaries using multiple metrics.

    Models are loaded once in @modal.enter() and reused across all
    evaluate() calls from any pipeline.

    Metrics:
        - BLEU: N-gram overlap
        - ROUGE-L: Longest common subsequence
        - SBERT Coherence: Semantic similarity
        - BERTScore F1: Contextual embedding similarity
        - scispaCy Entity Recall: CUI-based medical entity coverage
        - MedCAT Entity Recall: CUI-based medical entity coverage
    """

    @modal.enter()
    def load_models(self):
        """Load all evaluation models once when container starts."""
        from sentence_transformers import SentenceTransformer
        import nltk
        import spacy
        from scispacy.abbreviation import AbbreviationDetector
        from scispacy.linking import EntityLinker
        from medcat.cat import CAT

        print("🔄 Loading shared evaluation models...")

        # ==============================
        # 1. NLTK Data
        # ==============================
        print("  → Downloading NLTK data...")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        # ==============================
        # 2. SBERT for Coherence
        # ==============================
        print("  → Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✅ SBERT loaded!")

        # ==============================
        # 3. scispaCy + UMLS Linker
        # ==============================
        print("  → Loading scispaCy + UMLS linker...")
        self.scispacy_nlp = spacy.load("en_core_sci_scibert")
        self.scispacy_nlp.add_pipe("abbreviation_detector", first=True)
        self.scispacy_nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "k": 30,
            "threshold": 0.7,
            "no_definition_threshold": 0.95,
            "filter_for_definitions": True,
            "max_entities_per_mention": 1,
        })
        self.scispacy_linker = self.scispacy_nlp.get_pipe("scispacy_linker")
        print("  ✅ scispaCy + UMLS linker loaded!")

        # ==============================
        # 4. MedCAT
        # ==============================
        print("  → Loading MedCAT model...")
        medcat_model_path = "/medcat_models/medmen_wstatus_2021_oct.zip"
        self.medcat = CAT.load_model_pack(medcat_model_path)
        print("  ✅ MedCAT loaded!")

        print("✅ All shared evaluation models loaded!")

    # ==============================
    # Helper Methods for CUI Extraction
    # ==============================

    def _extract_scispacy_cuis(self, text: str) -> set:
        """Extract UMLS CUIs from text using scispaCy."""
        if not text or not text.strip():
            return set()

        doc = self.scispacy_nlp(text)
        cuis = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                cuis.add(ent._.kb_ents[0][0])  # Get top CUI
        return cuis

    def _extract_medcat_cuis(self, text: str) -> set:
        """Extract UMLS CUIs from text using MedCAT."""
        if not text or not text.strip():
            return set()

        result = self.medcat.get_entities(text)
        cuis = set()
        for ent_id, ent_data in result['entities'].items():
            cuis.add(ent_data['cui'])
        return cuis

    def _compute_cui_recall(self, reference_cuis: set, generated_cuis: set) -> float:
        """Compute CUI-based entity recall."""
        if not reference_cuis:
            return 0.0
        matched = reference_cuis.intersection(generated_cuis)
        return len(matched) / len(reference_cuis)

    # ==============================
    # Main Evaluation Method
    # ==============================

    @modal.method()
    def evaluate(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Evaluate generated summary against reference using multiple metrics.

        Args:
            generated: Generated summary text
            reference: Reference summary text

        Returns:
            dict with:
                - bleu: BLEU score
                - rouge_l: ROUGE-L F1 score
                - sbert_coherence: SBERT cosine similarity
                - bert_f1: BERTScore F1
                - scispacy_entity_recall: CUI-based recall (scispaCy)
                - medcat_entity_recall: CUI-based recall (MedCAT)
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from sentence_transformers import util
        from bert_score import score

        print("🔹 Computing evaluation metrics...")

        # Handle empty inputs
        if not generated or not generated.strip() or not reference:
            print("⚠️ Warning: Empty text, returning zero scores")
            return {
                "bleu": 0.0,
                "rouge_l": 0.0,
                "sbert_coherence": 0.0,
                "bert_f1": 0.0,
                "scispacy_entity_recall": 0.0,
                "medcat_entity_recall": 0.0,
            }

        # ==============================
        # EXISTING METRICS
        # ==============================

        # BLEU Score (with smoothing for short texts)
        print("  → Computing BLEU...")
        smoother = SmoothingFunction()
        bleu = sentence_bleu(
            [reference.split()],
            generated.split(),
            smoothing_function=smoother.method1
        )

        # ROUGE-L
        print("  → Computing ROUGE-L...")
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = scorer.score(reference, generated)["rougeL"].fmeasure

        # SBERT Coherence
        print("  → Computing SBERT coherence...")
        ref_emb = self.sbert_model.encode(reference, convert_to_tensor=True)
        gen_emb = self.sbert_model.encode(generated, convert_to_tensor=True)
        sbert_coherence = util.cos_sim(ref_emb, gen_emb).item()

        # BERTScore
        print("  → Computing BERTScore...")
        P, R, F1 = score([generated], [reference], lang="en", verbose=False)
        bert_f1 = F1.mean().item()

        # ==============================
        # CUI-BASED ENTITY RECALL
        # ==============================

        # scispaCy Entity Recall
        print("  → Computing scispaCy entity recall...")
        scispacy_ref_cuis = self._extract_scispacy_cuis(reference)
        scispacy_gen_cuis = self._extract_scispacy_cuis(generated)
        scispacy_entity_recall = self._compute_cui_recall(scispacy_ref_cuis, scispacy_gen_cuis)
        print(f"    scispaCy: {len(scispacy_ref_cuis)} ref CUIs, {len(scispacy_gen_cuis)} gen CUIs, recall={scispacy_entity_recall:.4f}")

        # MedCAT Entity Recall
        print("  → Computing MedCAT entity recall...")
        medcat_ref_cuis = self._extract_medcat_cuis(reference)
        medcat_gen_cuis = self._extract_medcat_cuis(generated)
        medcat_entity_recall = self._compute_cui_recall(medcat_ref_cuis, medcat_gen_cuis)
        print(f"    MedCAT: {len(medcat_ref_cuis)} ref CUIs, {len(medcat_gen_cuis)} gen CUIs, recall={medcat_entity_recall:.4f}")

        # ==============================
        # RESULTS
        # ==============================
        results = {
            "bleu": bleu,
            "rouge_l": rouge_l,
            "sbert_coherence": sbert_coherence,
            "bert_f1": bert_f1,
            "scispacy_entity_recall": scispacy_entity_recall,
            "medcat_entity_recall": medcat_entity_recall,
        }

        print(f"  ✅ BLEU: {bleu:.4f} | ROUGE-L: {rouge_l:.4f} | SBERT: {sbert_coherence:.4f} | BERTScore: {bert_f1:.4f}")
        print(f"  ✅ scispaCy Recall: {scispacy_entity_recall:.4f} | MedCAT Recall: {medcat_entity_recall:.4f}")

        return results


# ============================================================================
# Health Check / Test Endpoint
# ============================================================================

@app.function(image=evaluator_image)
def health_check() -> str:
    """Simple health check to verify service is running."""
    return "✅ Shared Evaluator Service is running!"


# ============================================================================
# Local Test Entrypoint
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the evaluator with sample data."""
    print("=" * 60)
    print("🧪 Testing Shared Evaluator Service")
    print("=" * 60)

    # Sample data
    reference = """
    Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disease 
    caused by the expansion of a polyglutamine tract within the androgen receptor (AR).
    """

    generated = """
    Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disorder 
    that results from an expanded polyglutamine tract in the androgen receptor (AR).
    """

    # Run evaluation
    evaluator = SummaryEvaluator()
    metrics = evaluator.evaluate.remote(generated, reference)

    # Display results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:<25}: {value:.4f}")

    print("\n✅ Test complete!")