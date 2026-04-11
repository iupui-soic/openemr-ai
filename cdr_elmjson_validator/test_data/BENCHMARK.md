# ELM Validation Benchmark

## Overview

This benchmark evaluates whether LLMs can verify that clinical decision support
(CDS) implementations in HL7 ELM JSON format correctly match their corresponding
Clinical Practice Guidelines (CPGs).

- **41 test cases** (16 valid, 25 invalid)
- **16 clinical domains** spanning USPSTF preventive screening plus ATS/ERS IPF
- Each test case pairs an ELM JSON file with a CPG reference document
- Ground truth labels in `ground_truth.json`
- Class distribution: 39.0% valid (16/41) / 61.0% invalid (25/41)
- Invalid cases split: **13 parametric errors** + **12 semantic logic errors**
  across 4 categories

## Test Case Construction

### Valid Cases (16)

Each valid ELM file was sourced from real CDS implementations (OpenEMR or
CDS Connect repositories) that correctly implement clinical screening
recommendations. The CPG markdown documents were extracted from the
corresponding USPSTF statements or (for IPF) the ATS/ERS/JRS/ALAT 2022 guideline.

**Clinical domains covered:**

- Cancer screening: breast, cervical, colon (OpenEMR), lung, prostate
- Infectious disease: chlamydia
- Cardiovascular: AAA
- Mental health: depression
- Substance use: alcohol, tobacco
- Musculoskeletal: osteoporosis
- Fall risk: falls prevention
- Weight: adult weight screening and follow-up
- Multi-library CQL: USPSTF statin shared logic, condition/medication count
- Pulmonary: idiopathic pulmonary fibrosis (IPF)

### Invalid Cases (25)

Invalid cases comprise two error categories that test distinct capabilities:

#### Parametric Errors (13 cases)

Constructed by modifying numeric values in correct ELM implementations to
create realistic but clinically meaningful discrepancies. This reflects the
most common real-world error mode when transcribing CPGs into CQL, as logical
nesting errors and syntax violations are typically caught by the CQL-to-ELM
compiler itself.

| Invalid Case | Error Type | Specific Mismatch |
|---|---|---|
| Hypertension-Screening | Age + time | Age >= 25 (should be 18); 6-month lookback (should be 1 year) |
| Statin-Therapy-for-CVD-Prevention | Age | Upper age 80 (should be 85) |
| Type-2-Diabetes-Diagnosis | Value | HbA1c >= 9.0% (should be >= 6.5%) |
| Colorectal-Cancer-Average-Risk | Age + time | Age >= 60 (should be 45); 2-year lookback (should be 1 year) |
| Anxiety-Screening | Time | 6-month lookback (should be 1 year) |
| HIV-Screening | Age | Age >= 30 (should be 15-65) |
| Prediabetes-Obesity-Screening | Value + time | BMI >= 35 (should be 25); 1-year lookback (should be 3 years) |
| Cervical-Cancer-Screening-WrongAge | Age | Age >= 30 (should be 21) |
| Falls-Prevention-WrongAge | Age | Age >= 75 (should be 65) |
| Alcohol-Misuse-WrongAge | Age | Age >= 25 (should be 18) |
| AAA-Screening-WrongAge | Age | Age >= 55 (should be 65) |
| Depression-Screening-WrongLookback | Time | 3-month lookback (should be 1 year) |
| Osteoporosis-WrongLookback | Time | 6-month lookback (should be 2 years) |

#### Semantic Logic Errors (12 cases, 4 categories)

Structural/logical modifications that compile correctly but implement the
wrong clinical logic. These test whether models can detect errors beyond
simple numeric mismatches, requiring understanding of Boolean operator
semantics, expression tree structure, and value set references.

**Category 1: Missing condition** — expression branch deleted from an AND chain

| Invalid Case | Specific Mismatch |
|---|---|
| AAA-Missing-SexRestriction | Male-sex restriction removed (female patients falsely qualify) |
| Depression-Missing-BipolarExclusion | NOT(bipolar diagnosis) exclusion removed |
| Tobacco-Missing-Exclusion | Prior-screening lookback removed entirely |
| Lung-Cancer-Missing-SubPopulation | Upper age limit removed (>=50 with no cap) |

**Category 2: Inverted logic** — NOT operator added or removed

| Invalid Case | Specific Mismatch |
|---|---|
| Colon-Cancer-Inverted-Procedure | Count==0 changed to Count>0 (screens patients who ALREADY had colonoscopy) |
| Falls-Inverted-Assessment | NOT removed from NOT(HasRecentAssessment) |
| IPF-Screening-WrongCT | Not-exists changed to exists for Connective Tissue Disorder exclusion |

**Category 3: Wrong nesting** — Boolean operators restructured

| Invalid Case | Specific Mismatch |
|---|---|
| Breast-Cancer-WrongOperator | AND changed to OR in inclusion criteria (female OR age>=40) |
| Depression-WrongNesting | NOT(A) AND NOT(B) changed to NOT(A AND B) (De Morgan's law violation) |
| IPF-WrongExclusionNesting | OR changed to AND in exclusion criteria (both CTD and amiodarone required to exclude) |

**Category 4: Swapped reference** — value set OID points to wrong clinical concept

| Invalid Case | Specific Mismatch |
|---|---|
| Falls-SwappedValueSet | Fall Risk Assessment VS replaced with Adult Depression Screening VS |
| Alcohol-SwappedValueSet | Alcohol Use Screening VS replaced with Tobacco Use Screening VS |

### Error Injection Methodology

- **Parametric errors**: Only numeric values modified (ages, time intervals,
  clinical thresholds). Each modification creates a clinically meaningful
  difference. Logical structure and expression nesting preserved.
- **Semantic errors**: Expression tree structure modified (Boolean operators
  swapped, expression branches deleted, NOT operators added/removed, value
  set OIDs replaced). These are valid ELM that the CQL compiler accepts but
  that implement incorrect clinical logic.
- Each invalid case uses the **same CPG file** as its source valid case —
  the error is in the ELM implementation, not the CPG reference.
- Parametric cases were generated via `create_expanded_cases.py` and verified
  manually.
- Semantic cases were generated via `create_semantic_cases.py` and verified
  manually. IPF-Screening-WrongCT was authored directly against the ATS/ERS
  guideline source.

### Class Distribution

- 39.0% positive (valid: 16/41) / 61.0% negative (invalid: 25/41)
- A naive "always valid" classifier achieves 39.0% accuracy
- A naive "always invalid" classifier achieves 61.0% accuracy
- Frontier models must substantially exceed both baselines to demonstrate
  genuine validation capability with balanced sensitivity and specificity

## File Structure

```
test_data/
├── ground_truth.json              # Labels, CPG mappings, expected errors (41 cases)
├── BENCHMARK.md                   # This file
├── AAA-Screening.json             # Valid ELM files (16)
├── AAA-Screening_CPG.md           # Corresponding CPG documents
├── AAA-Screening-WrongAge.json    # Parametric variant (reuses same CPG)
├── AAA-Missing-SexRestriction.json # Semantic variant (reuses same CPG)
├── Breast-Cancer-WrongOperator.json # Semantic logic error
├── IPF-Screening.json             # IPF valid case
├── IPF-Screening-WrongCT.json     # IPF semantic error
├── IPF-Screening_CPG.md           # IPF CPG from ATS/ERS guideline
├── ...
└── Alcohol-SwappedValueSet.json
```

## Ground Truth Schema

`ground_truth.json` contains:

```json
{
  "test_cases": {
    "<filename>.json": {
      "valid": true/false,
      "cpg_file": "<cpg_filename>.md",
      "expected_errors": ["keyword1", "keyword2"],
      "expected_warnings": [],
      "notes": "Description of what this case tests"
    }
  }
}
```

- `valid`: Whether the ELM correctly implements the CPG
- `cpg_file`: Reference CPG document for comparison
- `expected_errors`: Keywords that should appear in error explanations
  (for invalid cases). Used for error_match scoring.
- `notes`: Human-readable description. For semantic errors, includes
  the substring "semantic" for programmatic categorization.

## Evaluation Protocol

1. **Input**: ELM JSON is processed by the two-phase ELM Simplifier:
   - Phase 1: Extract age thresholds, time intervals, value set references
     as structured key-value pairs
   - Phase 2: Translate the full expression tree into human-readable logic
     summary preserving AND/OR/NOT operators, expression references, IsNull,
     Equivalent, In, temporal operators, and Query structure
   (implemented in `elm_simplifier.py::compare_format`)
2. **Prompt**: Simplified ELM + CPG presented to model with structured
   output instructions (VALID: YES/NO, ERRORS: list)
3. **Parsing**: Response parsed for VALID/ERRORS fields (supports both
   bare responses and responses wrapped in `<think>...</think>` tags)
4. **Scoring**: Binary accuracy (correct valid/invalid prediction) plus
   per-category disaggregation (parametric, semantic, valid)

### Metrics

| Metric | Definition |
|---|---|
| Accuracy | (TP + TN) / N |
| Sensitivity | TP / (TP + FN) — correctly identifies valid ELM |
| Specificity | TN / (TN + FP) — correctly identifies invalid ELM |
| F1 | 2·TP / (2·TP + FP + FN) |
| Parametric accuracy | Correct / 13 parametric cases |
| Semantic accuracy | Correct / 12 semantic cases |
| Valid accuracy | Correct / 16 valid cases |

### Statistical Tests

| Test | Purpose |
|---|---|
| Fisher's exact test | Group comparison (frontier vs mid-range, frontier vs small) |
| McNemar's exact test | Pairwise model comparison on matched cases (Bonferroni corrected) |
| Multi-trial variance | Mean ± SD over 5 independent trials at T=0.1 |

## Models Evaluated

### Open-Weight Frontier (7 models)

| Model | Params / Active | Infrastructure |
|---|---|---|
| Gemma 4 31B | 31B dense | Local RTX 6000 (4-bit quantization) |
| Gemma 4 26B A4B | 26B / 4B active MoE | Local RTX 6000 (4-bit) |
| Qwen3 32B | 32B dense | Groq API |
| GPT-OSS 120B | 120B / 5.1B active MoE | Groq API |
| Qwen3.5 35B A3B | 35B / 3B active MoE | OpenRouter API |
| Llama 3.3 70B | 70B dense | Groq API |
| GPT-OSS 20B | 20B / 3.6B active MoE | Groq API |

### Proprietary Reference (1 model)

| Model | Infrastructure |
|---|---|
| GPT-5.4 mini | OpenAI API (reasoning_effort=low) |

### Mid-Range (4 models, 3.8–4B)

| Model | Infrastructure |
|---|---|
| MedGemma 1.5 4B | Local RTX 6000 (bfloat16) |
| Phi-3 Mini | Local RTX 6000 |
| Gemma 3 4B | OpenRouter API |
| MedGemma 4B | Local RTX 6000 (bfloat16) |

### Small (4 models, 1–3B)

| Model | Infrastructure |
|---|---|
| Llama-3.2-1B | Local RTX 6000 |
| Qwen-2.5-3B | Local RTX 6000 |
| Llama-3.2-3B | Local RTX 6000 |
| Qwen-2.5-1.5B | Local RTX 6000 |

## Ablation Study Design

Four conditions tested across 8 frontier-class models (7 open-weight + 1
proprietary reference) with 5 trials per condition per model at temperature 0.1:

| Condition | ELM Input | CPG Reference |
|---|---|---|
| Full (baseline) | Simplified (both phases) | Yes |
| No CPG | Simplified (both phases) | No |
| No Simplification | Raw ELM JSON | Yes |
| Neither | Raw ELM JSON | No |

Results in `results/ablation_multi_trial/`.

## Limitations

1. **Sample size**: n=41 provides improved discriminative power over the
   earlier 31-case version, but fine-grained pairwise ranking of top-tier
   models (all >85% accuracy) still benefits from larger benchmarks. The
   primary finding (frontier vs mid-range gap, Fisher p<0.001) is robust.
2. **Error type coverage**: 13 parametric + 12 semantic errors across 4
   semantic categories (missing condition, inverted logic, wrong nesting,
   swapped reference). Does not exhaustively test terminology binding
   errors or complex multi-library interactions.
3. **Clinical domain**: Primarily USPSTF preventive screening plus IPF;
   other CDS domains (treatment protocols, drug-drug interactions,
   dosing logic) may show different performance characteristics.
4. **Deterministic simplification**: The ELM Simplifier's two-phase
   extraction is comprehensive for standard HL7 ELM constructs but may
   miss clinical logic expressed through unusual or custom ELM extensions.

## Reproducibility

All frontier, proprietary, and mid-range model results are means over
**5 independent trials at temperature 0.1** to quantify stochastic variance.
Multi-trial protocol (see `run_multi_trial.py` and `run_new_models_multi_trial.py`)
was applied uniformly across all 16 open-weight models and the GPT-5.4 mini
proprietary reference.

Scripts:

- `run_multi_trial.py` — 5-trial multi-trial for the 4 Groq-hosted frontier
  models (GPT-OSS 20B/120B, Llama 3.3 70B, Qwen3-32B)
- `run_ablation_multi_trial.py` — 4-condition × 5-trial ablation for the
  same 4 Groq models
- `run_new_models_multi_trial.py` — 5-trial multi-trial for Gemma 4 models,
  Qwen3.5, GPT-5.4 mini, MedGemma, and local small models
- `run_new_ablation_multi_trial.py` — 4-condition × 5-trial ablation for
  Gemma 4, Qwen3.5, and GPT-5.4 mini
- `create_expanded_cases.py` — regenerate parametric error cases
- `create_semantic_cases.py` — regenerate semantic error cases (categories
  1-4 excluding the original 3 and IPF-WrongCT)

Raw per-trial CSVs and summaries:

- `results/multi_trial/` — multi-trial results for all models
- `results/ablation_multi_trial/` — ablation results for 8 models
- `results/multi_trial/multi_trial_summary.csv` — aggregate statistics
- `results/multi_trial/per_case_stability.csv` — per-case stability across trials

## Reproducing Results

```bash
# Multi-trial for Groq frontier models (5 trials x 4 models x 41 cases)
GROQ_API_KEY=xxx python run_multi_trial.py

# Ablation for Groq frontier models (5 trials x 4 cond x 4 models x 41 cases)
GROQ_API_KEY=xxx python run_ablation_multi_trial.py

# Multi-trial for Gemma 4, Qwen3.5, GPT-5.4 mini, mid-range, small
HF_TOKEN=xxx OPENROUTER_API_KEY=xxx OPENAI_API_KEY=xxx \
    python run_new_models_multi_trial.py --model gemma-4-26b-a4b
#   ...repeat --model for each model, or omit --model to run all

# Ablation for new frontier models
HF_TOKEN=xxx OPENROUTER_API_KEY=xxx OPENAI_API_KEY=xxx \
    python run_new_ablation_multi_trial.py --model gemma-4-26b-a4b
```

## Analysis Notebooks

Pre-rendered Jupyter notebooks with all analysis outputs:

| Notebook | Contents |
|---|---|
| [`01_elm_validation_results.ipynb`](../notebooks/01_elm_validation_results.ipynb) | Main results across 16 models with per-category disaggregation, Fisher test, McNemar pairwise |
| [`02_error_analysis.ipynb`](../notebooks/02_error_analysis.ipynb) | Heatmap, case difficulty, error disaggregation, qualitative examples |
| [`03_ablation_study.ipynb`](../notebooks/03_ablation_study.ipynb) | Component contribution analysis across 8 frontier models |
| [`04_prompt_engineering.ipynb`](../notebooks/04_prompt_engineering.ipynb) | 5 strategies × 3 models comparison |
