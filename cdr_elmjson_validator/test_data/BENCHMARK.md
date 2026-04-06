# ELM Validation Benchmark

## Overview

This benchmark evaluates whether LLMs can verify that clinical decision support
(CDS) implementations in HL7 ELM JSON format correctly match their corresponding
Clinical Practice Guidelines (CPGs).

- **22 test cases** (15 valid, 7 invalid)
- **15 USPSTF screening interventions** spanning preventive care domains
- Each test case pairs an ELM JSON file with a CPG reference document
- Ground truth labels in `ground_truth.json`

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

### Invalid Cases (7)

Invalid cases were constructed by modifying one or more numeric values in
correct ELM implementations to create realistic but incorrect logic. Each
invalid case has a specific, documentable mismatch with its CPG.

| Invalid Case | Error Type | Specific Mismatch |
|---|---|---|
| Hypertension-Screening | Age + time | Age >= 25 (should be 18); 6-month lookback (should be 1 year) |
| Statin-Therapy-for-CVD-Prevention | Age | Upper age 80 (should be 85) |
| Type-2-Diabetes-Diagnosis | Value | HbA1c >= 9.0 g% (should be >= 6.5%) |
| Colorectal-Cancer-Average-Risk | Age + time | Age >= 60 (should be 45); 2-year lookback (should be 1 year) |
| Anxiety-Screening | Time | 6-month lookback (should be 1 year) |
| HIV-Screening | Age | Age >= 30 (should be 15-65) |
| Prediabetes-Obesity-Screening | Value + time | BMI >= 35 (should be 25); 1-year lookback (should be 3 years) |

**Error injection methodology:**
- Only numeric values were modified (ages, time intervals, clinical thresholds)
- Each modification creates a clinically meaningful difference (e.g., missing
  a screening population segment, wrong screening frequency)
- Logical structure and expression nesting were preserved
- Value sets and code references were not modified

### Class Distribution

- 68.2% positive (valid) / 31.8% negative (invalid)
- A majority-class baseline classifier achieves 68.2% accuracy
- Models must substantially exceed this to demonstrate validation capability

## File Structure

```
test_data/
├── ground_truth.json              # Labels, CPG mappings, expected errors
├── BENCHMARK.md                   # This file
├── AAA-Screening.json             # Valid ELM files
├── AAA-Screening_CPG.md           # Corresponding CPG documents
├── Alcohol-Misuse-Screening.json
├── Alcohol-Misuse-Screening_CPG.md
├── ...                            # (15 valid + 7 invalid = 22 pairs)
└── HIV-Screening_CPG.md
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
| Cohen's w / chi-square | Effect size vs base rate |
| Post-hoc power analysis | Detectable effect at n=22 |

## Limitations

1. **Sample size**: n=22 limits power to detect medium effects
   (need n>=88 for w=0.3 at 80% power)
2. **Error types**: Only numeric value mismatches — does not test
   logical errors, operator mistakes, or missing clinical pathways
3. **Single domain**: All cases are USPSTF preventive screening;
   other CDS domains (e.g., treatment protocols, drug interactions)
   may have different difficulty profiles
4. **Deterministic simplification**: The ELM simplifier extracts a
   fixed set of features; errors in unextracted features would not
   be detected by any model using this pipeline

## Reproducing Results

```bash
# Run a single model
python run_validation.py --model gpt-oss-20b --output results/results-gpt-oss-20b.csv

# Run all models
python run_validation.py --all-models --output-dir results/

# Run with ablation
python run_ablation.py --model gpt-oss-20b

# Run prompt experiments
python run_prompt_experiments.py --model gpt-oss-20b

# Analyze existing results
python analyze_elm_results.py
```
