# Prostate Cancer Screening Logic

## Target Population
Patients must meet **ALL** of the following criteria to be eligible for screening:
1.  **Age:** Patient is **50 years** or older.
2.  **Gender:** Patient gender is **Male**.
3.  **No Recent Screening:** Patient has **0 (zero)** documented screening procedures in the past **1 month**.
    * *Note:* The system checks for **any** of the following procedures:
        * PSA Test (VS `...3.526.2.215`)
        * Digital Rectal Exam (VS `...3.7643.3.1042`)
        * Prostate Biopsy (VS `...1.4.1078.1248`)

## Recommendation Rule
**IF** the patient is in the Target Population:
**THEN** the system must recommend:
* "Perform prostate cancer screening (PSA test, DRE, or biopsy as appropriate)"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the Target Population criteria:
**THEN** the system returns no specific error message (Null).