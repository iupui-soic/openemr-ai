"""
Text Evaluator Service - Handles text-based evaluation metrics.

This service provides evaluation metrics for medical summarization:
- BLEU: N-gram overlap
- ROUGE-L: Longest common subsequence
- SBERT Coherence: Semantic similarity
- BERTScore F1: Contextual embedding similarity

Usage:
    modal deploy text_evaluator_service.py

Calling from other pipelines:
    text_eval_app = modal.App.lookup("text-evaluator-service")
    TextEvaluator = text_eval_app.cls("TextEvaluator")
    evaluator = TextEvaluator()
    metrics = evaluator.evaluate.remote(generated_text, reference_text)
"""

import modal
from typing import Dict

# ============================================================================
# Modal App Configuration
# ============================================================================

app = modal.App("text-evaluator-service")

# ============================================================================
# Text Evaluator Image
# ============================================================================

text_evaluator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "transformers==4.44.0",
        "sentence-transformers==2.7.0",
        "bert-score>=0.3.13",
        "pandas>=2.0.0",
        "torch>=2.0.0",
    )
)


# ============================================================================
# Text Evaluator Class
# ============================================================================

@app.cls(
    image=text_evaluator_image,
    timeout=1800,
    gpu="T4",
    memory=8192,
    scaledown_window=600,
)
class TextEvaluator:
    """
    Text-based evaluator for medical summaries.

    Models are loaded once in @modal.enter() and reused across all
    evaluate() calls from any pipeline.

    Metrics:
        - BLEU: N-gram overlap
        - ROUGE-L: Longest common subsequence
        - SBERT Coherence: Semantic similarity
        - BERTScore F1: Contextual embedding similarity
    """

    @modal.enter()
    def load_models(self):
        """Load all evaluation models once when container starts."""
        from sentence_transformers import SentenceTransformer
        import nltk

        print("🔄 Loading text evaluation models...")

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

        print("✅ All text evaluation models loaded!")

    @modal.method()
    def evaluate(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Evaluate generated summary against reference using text-based metrics.

        Args:
            generated: Generated summary text
            reference: Reference summary text

        Returns:
            dict with:
                - bleu: BLEU score
                - rouge_l: ROUGE-L F1 score
                - sbert_coherence: SBERT cosine similarity
                - bert_f1: BERTScore F1
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from sentence_transformers import util
        from bert_score import score

        print("🔹 Computing text evaluation metrics...")

        # Handle empty inputs
        if not generated or not generated.strip() or not reference:
            print("⚠️ Warning: Empty text, returning zero scores")
            return {
                "bleu": 0.0,
                "rouge_l": 0.0,
                "sbert_coherence": 0.0,
                "bert_f1": 0.0,
            }

        # ==============================
        # BLEU Score
        # ==============================
        print("  → Computing BLEU...")
        smoother = SmoothingFunction()
        bleu = sentence_bleu(
            [reference.split()],
            generated.split(),
            smoothing_function=smoother.method1
        )

        # ==============================
        # ROUGE-L
        # ==============================
        print("  → Computing ROUGE-L...")
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = scorer.score(reference, generated)["rougeL"].fmeasure

        # ==============================
        # SBERT Coherence
        # ==============================
        print("  → Computing SBERT coherence...")
        ref_emb = self.sbert_model.encode(reference, convert_to_tensor=True)
        gen_emb = self.sbert_model.encode(generated, convert_to_tensor=True)
        sbert_coherence = util.cos_sim(ref_emb, gen_emb).item()

        # ==============================
        # BERTScore
        # ==============================
        print("  → Computing BERTScore...")
        P, R, F1 = score([generated], [reference], lang="en", verbose=False)
        bert_f1 = F1.mean().item()

        results = {
            "bleu": bleu,
            "rouge_l": rouge_l,
            "sbert_coherence": sbert_coherence,
            "bert_f1": bert_f1,
        }

        print(f"  ✅ BLEU: {bleu:.4f} | ROUGE-L: {rouge_l:.4f} | SBERT: {sbert_coherence:.4f} | BERTScore: {bert_f1:.4f}")

        return results


# ============================================================================
# Health Check / Test Endpoint
# ============================================================================

@app.function(image=text_evaluator_image)
def health_check() -> str:
    """Simple health check to verify service is running."""
    return "✅ Text Evaluator Service is running!"


# ============================================================================
# Local Test Entrypoint
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the evaluator with sample data."""
    print("=" * 60)
    print("🧪 Testing Text Evaluator Service")
    print("=" * 60)

    reference = """
    Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disease 
    caused by the expansion of a polyglutamine tract within the androgen receptor (AR).
    """

    generated = """
    Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disorder 
    that results from an expanded polyglutamine tract in the androgen receptor (AR).
    """

    evaluator = TextEvaluator()
    metrics = evaluator.evaluate.remote(generated, reference)

    print("\n" + "=" * 60)
    print("📊 TEXT EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"  {metric:<20}: {value:.4f}")

    print("\n✅ Test complete!")