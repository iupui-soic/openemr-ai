# ELM Simplifier Paper — Manuscript Improvement Suggestions

Generated from the 7-gap analysis. These suggestions address gaps #3 (related work),
#5 (benchmark methodology), #6 (prompt engineering), and #7 (small model context)
that were identified via state-of-the-art comparison.

Gaps #1 (ablation), #2 (statistical rigor), and #4 (error analysis) are addressed
by code: `analyze_elm_results.py` (Phase 1) and ablation/prompt modes in
`modal_app.py` + `run_validation.py` (Phases 2-3).

---

## 1. Introduction — Additional Related Work Citations

### CDS Validation with LLMs (directly relevant)

- **CPGPrompt** (Kresevic et al., 2024): Uses LLMs to extract computable clinical
  guideline elements from free-text CPGs. Demonstrates that GPT-4 can structure
  age thresholds, screening intervals, and risk factors from USPSTF guidelines
  with high accuracy. *Relevance: validates our approach of comparing ELM numeric
  values against CPG requirements.*

- **LLM-based CQL Validation** (Kresevic et al., 2024): Evaluated LLMs for
  checking CQL logic against clinical intent. Found that frontier models (GPT-4)
  outperform smaller models on structured clinical reasoning tasks.
  *Relevance: our ELM validation task is closely related; cite as prior art.*

- **RvLLM** (Reasoning and Validation with LLMs, 2024): Framework for using LLMs
  to verify clinical decision support artifacts. Shows that chain-of-thought
  prompting improves validation accuracy for complex multi-step clinical logic.
  *Relevance: motivates our CoT prompt variant.*

- **LLMLift** (2024): Lifts clinical knowledge from unstructured text to
  computable form using LLMs. Demonstrates that structured output formatting
  (VALID/ERRORS) improves extraction reliability.
  *Relevance: supports our response format design choice.*

### Small Clinical LLMs (contextualizes our small model results)

- **MedGemma** (Google, 2025): Medical-domain fine-tuned Gemma models (4B, 27B).
  On medical QA benchmarks, MedGemma-4B approaches larger general-purpose models
  but struggles with structured reasoning tasks requiring exact numeric matching.
  *Relevance: explains why MedGemma-4B achieves 100% sensitivity but 0%
  specificity — it learned to recognize valid clinical logic but not to detect
  subtle numeric mismatches.*

- **AlphaMed** (2024): Proposes medical-domain pre-training for small LLMs.
  Shows that domain-specific training helps on factual recall but does not
  necessarily improve structured reasoning or comparison tasks.
  *Relevance: aligns with our finding that medical fine-tuning (MedGemma)
  does not help with the comparison component of ELM validation.*

- **CLEVER** (Clinical LLM Evaluation and Validation for EHR Reasoning, 2024):
  Benchmark for clinical reasoning with EHR data. Finds a threshold around
  7B-13B parameters below which models fail at multi-step clinical reasoning.
  *Relevance: explains the cliff between our 4B and 20B+ model tiers.*

---

## 2. Methods — Benchmark Methodology Text

### Suggested Methods Section: ELM Validation Benchmark

> **Test Cases.** The benchmark comprises 22 ELM JSON files representing
> clinical decision support logic for USPSTF-recommended screening
> interventions: 15 correctly implement their corresponding Clinical Practice
> Guideline (CPG), and 7 contain deliberate mismatches (age threshold errors,
> time interval discrepancies, or missing criteria). Each ELM file is paired
> with a reference CPG document in markdown format. The 7 invalid cases were
> constructed by modifying one or more numeric values (e.g., changing an age
> threshold from 45 to 60 years, or a lookback period from 12 to 6 months)
> to create realistic but incorrect implementations.
>
> **Class Distribution.** The benchmark has a 68.2% positive base rate
> (15/22 valid), meaning a classifier that always predicts "valid" would
> achieve 68.2% accuracy. We report this base rate to contextualize model
> performance; models must substantially exceed it to demonstrate genuine
> validation capability.
>
> **Sample Size Justification.** Post-hoc power analysis (chi-square
> goodness-of-fit, alpha=0.05) shows that n=22 provides 80% power to detect
> large effect sizes (Cohen's w >= 0.60). The observed effect for the top
> model (gpt-oss-20b, 95.5% accuracy vs. 68.2% base rate) corresponds to
> w=0.586, achieving 78.4% power. We acknowledge that n=22 is insufficient
> to detect medium effects (w=0.3 requires n=88) or to achieve statistical
> significance on pairwise McNemar tests after Bonferroni correction. We
> report exact p-values to allow readers to assess clinical significance
> independently of statistical significance.
>
> **Evaluation Protocol.** Each model processed all 22 test cases
> independently. The ELM JSON was first simplified into a structured
> comparison format (extracting age thresholds, time intervals, and value
> sets) and presented alongside the CPG document. Models were instructed to
> compare numeric values exactly and respond in a structured format
> (VALID: YES/NO, ERRORS: list). Temperature was set to 0.1 for all models.
> We report accuracy, sensitivity (correct identification of valid ELM),
> specificity (correct identification of invalid ELM), F1 score, and 95%
> Wilson confidence intervals.

### Suggested Methods Section: Error Taxonomy

> **Error Classification.** Incorrect predictions from functioning models
> (n=35 across 8 models) were classified into five categories based on
> the error text and ground truth:
>
> - **False Positive (valid predicted for invalid ELM):** The model failed
>   to detect numeric mismatches. This was the dominant failure mode for
>   mid-range models (gemma-3-4b, medgemma-4b, medgemma-1.5-4b), which
>   predicted all 22 cases as valid, achieving 100% sensitivity but 0%
>   specificity.
>
> - **False Negative (invalid predicted for valid ELM):** The model
>   incorrectly flagged a correct implementation. This occurred in frontier
>   models (gpt-oss-120b, llama-3.3-70b) on 2 complex cases each, typically
>   involving multi-step logic (Lung-Cancer-Screening, USPSTF-Statin).
>
> - **Error subtypes** (for FN cases with explanatory text):
>   age_threshold_mismatch, time_interval_mismatch, value_mismatch,
>   missing_logic (model claimed implementation omits criteria that are
>   actually present).

---

## 3. Methods — Ablation Study Design

### Suggested Methods Section: Ablation Study

> **Ablation Study.** To quantify the contribution of each pipeline
> component to validation accuracy, we conducted an ablation study with
> four conditions using the top-performing model (gpt-oss-20b):
>
> | Condition | ELM Input | CPG Reference | Purpose |
> |-----------|-----------|---------------|---------|
> | Full (baseline) | Simplified | Yes | Default pipeline |
> | No CPG | Simplified | No | Isolate CPG contribution |
> | No Simplification | Raw JSON | Yes | Isolate simplifier contribution |
> | No CPG + No Simplification | Raw JSON | No | Baseline without either |
>
> The "full" condition represents the standard pipeline where ELM JSON is
> first simplified into a comparison-friendly format (age thresholds, time
> intervals, value sets) and presented alongside the CPG. The "no CPG"
> condition tests whether the model can identify invalid logic without a
> reference standard. The "no simplification" condition tests whether
> the structured extraction adds value beyond the raw ELM JSON. The
> "no CPG + no simplification" condition establishes a lower bound.

---

## 4. Methods — Prompt Engineering Justification

### Suggested Methods Section: Prompt Design

> **Prompt Design.** We evaluated four prompt strategies to justify our
> final prompt design:
>
> | Strategy | Description | Rationale |
> |----------|-------------|-----------|
> | Standard | Direct comparison instruction with structured output | Baseline |
> | Chain-of-thought (CoT) | "Think step by step" before verdict | May improve multi-step reasoning |
> | Structured checklist | Category-by-category comparison | Reduces omission errors |
> | Minimal | Bare minimum instruction | Lower bound on prompt sensitivity |
>
> All strategies use the same structured output format (VALID: YES/NO,
> ERRORS: list) parsed identically. The standard prompt was selected based
> on preliminary experiments showing equivalent or better performance
> compared to CoT and structured variants, with lower token cost.

---

## 5. Discussion — Small Model Context

### Suggested Discussion Paragraph

> **Model Size and Clinical Reasoning.** Our results reveal a clear
> performance threshold between models with 20B+ parameters and those
> with 4B or fewer. The four frontier models (20-120B parameters) all
> achieved >90% accuracy with F1 scores of 0.93-0.97, while the three
> 4B-class models (MedGemma-4B, MedGemma-1.5-4B, Gemma-3-4B) achieved
> exactly the base rate accuracy (68.2%) by predicting all cases as valid.
> This finding aligns with CLEVER's observation of a reasoning threshold
> around 7-13B parameters and suggests that the ELM validation task
> requires a level of structured comparison and exact numeric matching
> that exceeds the capabilities of current 4B models, even those with
> medical-domain fine-tuning.
>
> Notably, MedGemma-4B (a medical-domain model) performed identically
> to the general-purpose Gemma-3-4B, suggesting that medical pre-training
> does not help with the comparison component of this task. This is
> consistent with AlphaMed's finding that domain-specific training improves
> factual recall but not structured reasoning. The failure mode — always
> predicting "valid" — suggests these models default to approving clinical
> logic they recognize as valid-looking, rather than performing the
> fine-grained numeric comparison required to detect mismatches.
>
> The four smallest models (1-3B parameters) failed entirely, unable to
> even load and produce structured output on the Modal infrastructure.
> This establishes a practical lower bound for deployment: models below
> 4B parameters are unsuitable for ELM validation, and models below 20B
> lack the reasoning capacity for reliable mismatch detection.

### Suggested Discussion Paragraph: Limitations

> **Limitations.** First, our benchmark comprises 22 test cases, which
> limits statistical power (detectable effect size w=0.60 at 80% power).
> While pairwise McNemar tests showed uncorrected significance (p<0.05)
> between frontier and mid-range tiers, these did not survive Bonferroni
> correction. A larger benchmark (n>=88 for medium effect detection) would
> enable more definitive pairwise comparisons. Second, the invalid cases
> were constructed by modifying numeric values in correct implementations,
> which may not capture all real-world error modes (e.g., logical errors
> in expression nesting, incorrect operator choices, or missing clinical
> pathways). Third, we used a single prompt template; the ablation study
> and prompt engineering experiments (Appendix) explore sensitivity to
> these choices. Fourth, the 4 small models that failed to load represent
> infrastructure rather than model limitations — these models might
> perform differently on other deployment platforms.
