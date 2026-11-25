"""
Entity Evaluator Service - Handles medical entity extraction and evaluation.

This service provides CUI-based entity metrics (recall, precision, F1) from:
- scispaCy: UMLS CUI-based medical entity coverage
- MedCAT: UMLS CUI-based medical entity coverage

Usage:
    modal deploy entity_evaluator_service.py

Calling from other pipelines:
    entity_eval_app = modal.App.lookup("entity-evaluator-service")
    EntityEvaluator = entity_eval_app.cls("EntityEvaluator")
    evaluator = EntityEvaluator()
    metrics = evaluator.evaluate.remote(generated_text, reference_text)
"""

import modal
from typing import Dict

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("entity-evaluator-service")

# ============================================================================
# Entity Evaluator Image
# ============================================================================

entity_evaluator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update && apt-get install -y wget unzip")
    .pip_install("numpy==1.26.4")
    .pip_install("spacy==3.7.5")
    .pip_install("scispacy==0.5.4")
    .run_commands(
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz"
    )
    .pip_install(
        "transformers==4.57.3",
        "peft==0.18.0",
        "medcat==1.16.7",
    )
    .run_commands(
        "mkdir -p /medcat_models",
        "wget -q https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip -O /medcat_models/medmen_wstatus_2021_oct.zip",
    )
)


# ============================================================================
# Entity Evaluator Class
# ============================================================================

@app.cls(
    image=entity_evaluator_image,
    timeout=1800,
    gpu="T4",
    memory=16384,
    scaledown_window=600,
)
class EntityEvaluator:
    """
    Entity-based evaluator for medical summaries using scispaCy and MedCAT.

    Models are loaded once in @modal.enter() and reused across all
    evaluate() calls from any pipeline.

    Metrics:
        - scispaCy Entity Recall: CUI-based medical entity coverage
        - MedCAT Entity Recall: CUI-based medical entity coverage
    """

    @modal.enter()
    def load_models(self):
        """Load all entity extraction models once when container starts."""
        import spacy
        from scispacy.abbreviation import AbbreviationDetector
        from scispacy.linking import EntityLinker
        from medcat.cat import CAT

        print("🔄 Loading entity evaluation models...")

        # ==============================
        # 1. scispaCy + UMLS Linker
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
        # 2. MedCAT
        # ==============================
        print("  → Loading MedCAT model...")
        medcat_model_path = "/medcat_models/medmen_wstatus_2021_oct.zip"
        self.medcat = CAT.load_model_pack(medcat_model_path)
        print("  ✅ MedCAT loaded!")

        print("✅ All entity evaluation models loaded!")

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

    def _compute_cui_prf(self, reference_cuis: set, generated_cuis: set) -> Dict[str, float]:
        """
        Recall, precision, and F1 over sets of unique CUIs.

        N = unique CUIs in the generated note, R = unique CUIs in the reference.
        recall = |N & R| / |R|, precision = |N & R| / |N|, F1 = harmonic mean.
        """
        matched = len(reference_cuis & generated_cuis)
        recall = matched / len(reference_cuis) if reference_cuis else 0.0
        precision = matched / len(generated_cuis) if generated_cuis else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "n_note_cuis": len(generated_cuis),
            "n_ref_cuis": len(reference_cuis),
            "n_matched_cuis": matched,
        }

    @modal.method()
    def evaluate(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Evaluate generated summary against reference using entity-based metrics.

        Args:
            generated: Generated summary text
            reference: Reference summary text

        Returns:
            dict with, for each tool in {scispacy, medcat}:
                - {tool}_entity_recall, {tool}_entity_precision, {tool}_entity_f1
                - {tool}_n_note_cuis, {tool}_n_ref_cuis, {tool}_n_matched_cuis
        """
        print("🔹 Computing entity evaluation metrics...")

        results = {}

        # Handle empty inputs
        if not generated or not generated.strip() or not reference:
            print("⚠️ Warning: Empty text, returning zero scores")
            for tool in ("scispacy", "medcat"):
                for k in ("recall", "precision", "f1"):
                    results[f"{tool}_entity_{k}"] = 0.0
                for k in ("n_note_cuis", "n_ref_cuis", "n_matched_cuis"):
                    results[f"{tool}_{k}"] = 0
            return results

        for tool, extract in (("scispacy", self._extract_scispacy_cuis),
                              ("medcat", self._extract_medcat_cuis)):
            print(f"  → Computing {tool} entity metrics...")
            ref_cuis = extract(reference)
            gen_cuis = extract(generated)
            prf = self._compute_cui_prf(ref_cuis, gen_cuis)
            for k in ("recall", "precision", "f1"):
                results[f"{tool}_entity_{k}"] = prf[k]
            for k in ("n_note_cuis", "n_ref_cuis", "n_matched_cuis"):
                results[f"{tool}_{k}"] = prf[k]
            print(f"    {tool}: {prf['n_ref_cuis']} ref CUIs, {prf['n_note_cuis']} gen CUIs, "
                  f"recall={prf['recall']:.4f} precision={prf['precision']:.4f} f1={prf['f1']:.4f}")

        return results


# ============================================================================
# Health Check / Test Endpoint
# ============================================================================

@app.function(image=entity_evaluator_image)
def health_check() -> str:
    """Simple health check to verify service is running."""
    return "✅ Entity Evaluator Service is running!"


# ============================================================================
# Local Test Entrypoint
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the evaluator with sample data."""
    print("=" * 60)
    print("🧪 Testing Entity Evaluator Service")
    print("=" * 60)

    reference = """
    Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disease 
    caused by the expansion of a polyglutamine tract within the androgen receptor (AR).
    """

    generated = """
    Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disorder 
    that results from an expanded polyglutamine tract in the androgen receptor (AR).
    """

    evaluator = EntityEvaluator()
    metrics = evaluator.evaluate.remote(generated, reference)

    print("\n" + "=" * 60)
    print("📊 ENTITY EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:<28}: {value:.4f}")

    print("\n✅ Test complete!")