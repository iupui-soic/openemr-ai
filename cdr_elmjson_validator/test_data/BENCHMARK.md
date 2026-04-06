# ELM Validation Benchmark

## Overview

This benchmark evaluates whether LLMs can verify that clinical decision support
(CDS) implementations in HL7 ELM JSON format correctly match their corresponding
Clinical Practice Guidelines (CPGs).

- **31 test cases** (15 valid, 16 invalid)
- **15 USPSTF screening interventions** spanning preventive care domains
- Each test case pairs an ELM JSON file with a CPG reference document
- Ground truth labels in `ground_truth.json`
- Near-balanced class distribution (48.4% valid / 51.6% invalid)

## Test Case Construction

### Valid Cases (15)

Each valid ELM file was sourced from real CDS implementations (OpenEMR or
CDS Connect repositories) that correctly implement USPSTF screening
recommendations. The CPG markdown documents were extracted from the
corresponding USPSTF recommendation statements.

**Screening domains covered:**
- Cancer screening: breast, cervical, colorectal (OpenEMR), lung, prostate
- Infectious disease: chlamydia, HIV (used as invalid case base)
- Metabolic: diabetes (used as invalid case base), obesity/prediabetes (invalid)
- Cardiovascular: hypertension (invalid), statin therapy (invalid), AAA
- Mental health: depression, anxiety (invalid)
- Substance use: alcohol, tobacco
- Musculoskeletal: osteoporosis
- Other: falls prevention, weight screening, USPSTF statin shared logic

### Invalid Cases (16)

Invalid cases comprise two error categories:

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
| Type-2-Diabetes-Diagnosis | Value | HbA1c >= 9.0 g% (should be >= 6.5%) |
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

#### Semantic Logic Errors (3 cases)

Structural/logical modifications that compile correctly but implement the
wrong clinical logic. These test whether models can detect errors beyond
simple numeric mismatches.

| Invalid Case | Error Type | Specific Mismatch |
|---|---|---|
| Breast-Cancer-WrongOperator | Operator swap | AND changed to OR in inclusion criteria (female OR age>=40, instead of female AND age>=40) |
| Tobacco-Missing-Exclusion | Missing check | Prior screening lookback removed entirely (all 18+ flagged regardless of prior screening) |
| Lung-Cancer-Missing-SubPopulation | Missing bound | Upper age limit removed (>=50 with no cap, should be 50-80) |

### Error Injection Methodology

- **Parametric errors**: Only numeric values modified (ages, time intervals,
  clinical thresholds). Each modification creates a clinically meaningful
  difference. Logical structure and expression nesting preserved.
- **Semantic errors**: Expression tree structure modified (Boolean operators
  swapped, expression branches deleted, comparison operands removed). These
  are valid ELM that the CQL compiler accepts but that implement incorrect
  clinical logic.
- Each invalid case uses the **same CPG file** as its source valid case —
  the error is in the ELM implementation, not the CPG reference.
- New cases were generated programmatically via `create_expanded_cases.py`
  and verified manually.

### Class Distribution

- 48.4% positive (valid) / 51.6% negative (invalid)
- Near-balanced distribution eliminates the class imbalance concern present
  in earlier versions (68.2% base rate)
- A naive "always valid" classifier now achieves only 48.4% accuracy
- Models must substantially exceed 50% to demonstrate validation capability

## File Structure

```
test_data/
├── ground_truth.json              # Labels, CPG mappings, expected errors (31 cases)
├── BENCHMARK.md                   # This file
├── AAA-Screening.json             # Valid ELM files (15)
├── AAA-Screening_CPG.md           # Corresponding CPG documents (15)
├── AAA-Screening-WrongAge.json    # Parametric variant (reuses same CPG)
├── Breast-Cancer-WrongOperator.json # Semantic logic error (reuses same CPG)
├── ...
└── Tobacco-Missing-Exclusion.json
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
- `notes`: Human-readable description of the test case and any
  intentional errors

## Evaluation Protocol

1. **Input**: ELM JSON is simplified into a structured format extracting
   age thresholds, time intervals, and value sets (via `simplify_elm_for_prompt`)
2. **Prompt**: Simplified ELM + CPG presented to model with structured
   output instructions (VALID: YES/NO, ERRORS: list)
3. **Parsing**: Response parsed for VALID/ERRORS fields
4. **Scoring**: Binary accuracy (correct valid/invalid prediction)

### Metrics

| Metric | Definition |
|---|---|
| Accuracy | (TP + TN) / N |
| Sensitivity | TP / (TP + FN) — correctly identifies valid ELM |
| Specificity | TN / (TN + FP) — correctly identifies invalid ELM |
| PPV | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| F1 | 2·TP / (2·TP + FP + FN) |
| Wilson CI | 95% confidence interval for accuracy proportion |

### Statistical Tests

| Test | Purpose |
|---|---|
| Wilson score interval | CIs for per-model accuracy (handles small n) |
| McNemar's exact test | Pairwise model comparison on matched cases |
| Fisher's exact test | Group comparison (frontier vs small models) |
| Cohen's w / chi-square | Effect size vs base rate |
| Post-hoc power analysis | Detectable effect at n=31 |

## Ablation Study Design

Four conditions tested across 4 frontier models (GPT-OSS-20B, GPT-OSS-120B,
Qwen3-32B, Llama 3.3 70B):

| Condition | ELM Input | CPG Reference |
|-----------|-----------|---------------|
| Full (baseline) | Simplified | Yes |
| No CPG | Simplified | No |
| No Simplification | Raw JSON | Yes |
| Neither | Raw JSON | No |

Results in `results/ablation/`.

## Prompt Engineering Design

Five strategies tested across 3 frontier models:

| Strategy | Description |
|----------|-------------|
| Standard | Direct comparison instruction |
| Chain-of-thought | Step-by-step reasoning before verdict |
| Few-shot | 2 exemplars (1 valid, 1 invalid) prepended |
| Structured | Category-by-category checklist |
| Minimal | Bare minimum instruction |

Results in `results/prompts/`.

## Limitations

1. **Sample size**: n=31 provides 80% power at Cohen's w>=0.50 (large
   effect). The primary finding (frontier vs small gap, Fisher p<0.001,
   OR=8.54) is robust. Fine-grained pairwise ranking needs n>=88.
2. **Error types**: Primarily parametric errors (numeric mismatches) with
   3 semantic logic errors. Does not test terminology binding errors or
   complex multi-library interactions.
3. **Single domain**: All cases are USPSTF preventive screening; other
   CDS domains (treatment protocols, drug interactions) may differ.
4. **Deterministic simplification**: The ELM simplifier extracts a fixed
   set of features; errors in unextracted features would not be detected.

## Reproducing Results

```bash
# Run a single model
python run_validation.py --model gpt-oss-20b --output results/results-gpt-oss-20b.csv

# Run all models (Groq API)
GROQ_API_KEY=xxx python run_experiments_direct.py --model gpt-oss-20b

# Run small models locally (requires GPU)
HF_TOKEN=xxx python run_small_models_local.py

# Run all models (Groq + local)
python run_all_expanded.py

# Run ablation study
python run_ablation.py --model gpt-oss-20b

# Run prompt experiments
python run_prompt_experiments.py --model gpt-oss-20b

# Run statistical analysis
python analyze_elm_results.py
```

## Analysis Notebooks

Pre-rendered Jupyter notebooks with all analysis outputs:

| Notebook | Contents |
|----------|----------|
| [`01_elm_validation_results.ipynb`](../notebooks/01_elm_validation_results.ipynb) | Main results, Wilson CIs, Fisher test, McNemar pairwise |
| [`02_error_analysis.ipynb`](../notebooks/02_error_analysis.ipynb) | Heatmap, case difficulty, error disaggregation, qualitative examples |
| [`03_ablation_study.ipynb`](../notebooks/03_ablation_study.ipynb) | Component contribution analysis across 4 models |
| [`04_prompt_engineering.ipynb`](../notebooks/04_prompt_engineering.ipynb) | 5 strategies x 3 models comparison |
