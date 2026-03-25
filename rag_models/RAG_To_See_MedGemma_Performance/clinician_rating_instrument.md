# Clinician Rating Instrument for AI-Generated Medical Summaries

## Study: RAG-Based Clinical Summarization Quality Evaluation

### Background

You are being asked to evaluate AI-generated medical summaries. Each summary was generated from a doctor-patient conversation transcript paired with electronic health record (EHR) data using a Retrieval-Augmented Generation (RAG) pipeline. The summaries follow a SOAP (Subjective, Objective, Assessment, Plan) format.

You will be provided with:
1. The **original transcript** (doctor-patient conversation)
2. The **OpenEMR data extract** (demographics, vitals, labs, medications, history)
3. The **AI-generated summary** (model identity is blinded)

### Rating Dimensions

Rate each summary on the following 6 dimensions using a 5-point Likert scale.

---

## Dimension 1: Accuracy

**Definition:** The information in the summary is factually consistent with the transcript and EHR data. No fabricated, contradictory, or misleading information is present.

| Score | Description |
|-------|-------------|
| 1 | Major factual errors — contains fabricated information or significantly contradicts source data |
| 2 | Several factual errors — some information does not match the transcript or EHR |
| 3 | Minor factual errors — mostly accurate but with a few inconsistencies |
| 4 | Nearly accurate — at most one trivial inaccuracy |
| 5 | Fully accurate — all information is consistent with the transcript and EHR data |

---

## Dimension 2: Completeness

**Definition:** All clinically relevant information from the transcript is captured in the summary. Important findings, symptoms, history, and clinical decisions are included.

| Score | Description |
|-------|-------------|
| 1 | Major omissions — multiple clinically important elements are missing |
| 2 | Significant omissions — several relevant clinical details are absent |
| 3 | Moderate completeness — captures main points but misses some relevant details |
| 4 | Nearly complete — almost all clinically relevant information is included |
| 5 | Comprehensive — all clinically relevant information from the transcript and EHR is captured |

---

## Dimension 3: Organization

**Definition:** The summary follows a logical SOAP structure with clear sections, appropriate grouping of information, and smooth transitions between sections.

| Score | Description |
|-------|-------------|
| 1 | Disorganized — no clear structure, information scattered without logical grouping |
| 2 | Poorly organized — some structure present but major sections are missing or misplaced |
| 3 | Moderately organized — recognizable structure but some sections are poorly delineated |
| 4 | Well organized — clear sections with minor organizational issues |
| 5 | Excellently organized — clear SOAP structure, logical flow, well-delineated sections |

---

## Dimension 4: Conciseness

**Definition:** The summary is appropriately concise without unnecessary repetition, redundancy, or verbosity. Information is presented efficiently.

| Score | Description |
|-------|-------------|
| 1 | Extremely verbose — excessive repetition, irrelevant details, or padding |
| 2 | Verbose — notable redundancy or unnecessary elaboration in multiple sections |
| 3 | Moderately concise — some repetition or verbose sections but generally reasonable |
| 4 | Concise — efficiently written with only minor redundancies |
| 5 | Optimally concise — no unnecessary repetition, every sentence adds value |

---

## Dimension 5: Clinical Utility

**Definition:** The summary would be useful for a clinician reviewing this patient's case. It provides actionable information and supports clinical decision-making.

| Score | Description |
|-------|-------------|
| 1 | Not useful — would not help a clinician understand or manage this patient |
| 2 | Minimally useful — provides some information but insufficient for clinical decisions |
| 3 | Moderately useful — provides a reasonable overview but lacks detail for confident decision-making |
| 4 | Very useful — provides most information needed for clinical review and management |
| 5 | Highly useful — excellent clinical summary that fully supports patient care decisions |

---

## Dimension 6: Overall Quality

**Definition:** Your overall assessment of the summary's quality as a clinical document. Consider all previous dimensions holistically.

| Score | Description |
|-------|-------------|
| 1 | Poor — significant issues across multiple dimensions; would not use in clinical practice |
| 2 | Below average — notable weaknesses; would require substantial revision |
| 3 | Average — acceptable quality with room for improvement |
| 4 | Good — high quality with only minor areas for improvement |
| 5 | Excellent — publication-quality clinical summary; would use as-is |

---

## Rating Protocol

1. **Read the original transcript** carefully to understand the clinical encounter
2. **Review the OpenEMR data extract** for objective clinical data (labs, vitals, medications, history)
3. **Read the AI-generated summary** without knowing which model produced it
4. **Rate each of the 6 dimensions** independently on the 1-5 scale
5. **Optional:** Add free-text comments for any summary (especially for scores of 1-2)

### Important Guidelines

- Rate each dimension **independently** — a summary may be well-organized (5) but incomplete (2)
- Compare the summary to the **source materials only** (transcript + EHR), not to your own expected summary
- A score of 3 means "acceptable/average" — use the full scale range
- If information is correctly identified as "not available" in the summary, that is NOT an error
- Focus on **clinical significance** — minor formatting issues should not heavily affect scores
- Summaries are presented in **randomized order** — do not try to identify the model

### Rating Sheet Format

For each summary, record:

```
Summary ID: [auto-assigned]
Rater ID: [your ID]

Accuracy:        [ 1 | 2 | 3 | 4 | 5 ]
Completeness:    [ 1 | 2 | 3 | 4 | 5 ]
Organization:    [ 1 | 2 | 3 | 4 | 5 ]
Conciseness:     [ 1 | 2 | 3 | 4 | 5 ]
Clinical Utility: [ 1 | 2 | 3 | 4 | 5 ]
Overall Quality:  [ 1 | 2 | 3 | 4 | 5 ]

Comments (optional): _______________
```

---

## Workload Estimate

- **Total summaries per rater:** 160 (40 conversations x 4 models)
- **Estimated time per summary:** 3-5 minutes
- **Total time:** 8-13 hours per rater
- **Recommended schedule:** 3-4 sessions of 2-3 hours over 1-2 weeks
- **Break recommendation:** Take a 10-minute break every 15-20 summaries

---

## References

This instrument is adapted from:
- **PDQI-9** (Physician Documentation Quality Instrument): Stetson PD, et al. "Assessing Electronic Note Quality Using the Physician Documentation Quality Instrument (PDQI-9)." Applied Clinical Informatics, 2012.
- **QNOTE** (Quality of Physician Notes): Hanson JL, et al. "Quality of outpatient clinical notes: a stakeholder definition derived through qualitative research." BMC Health Services Research, 2019.

---

## Data Collection Plan

### Rater Information
- **Number of raters:** 3
- **Rater profiles:** Physician fellows from distinct specialties (GI, IR, ER)
- **Blinding:** All raters are blinded to model identity

### Randomization
- Summaries are presented in a randomized order unique to each rater
- The same conversation's 4 model outputs are NOT presented consecutively
- Randomization seed is recorded for reproducibility

### Statistical Analysis
- **Inter-rater reliability:** Krippendorff's alpha (ordinal scale)
- **Supplementary reliability:** Intraclass Correlation Coefficient (ICC, two-way random)
- **Model comparison:** Friedman test (non-parametric repeated measures)
- **Pairwise comparison:** Wilcoxon signed-rank with Bonferroni correction (alpha = 0.05/6 = 0.0083)
- **Effect size:** Kendall's W
- **Target reliability:** Krippendorff's alpha >= 0.667 (acceptable), ideally >= 0.80 (good)
