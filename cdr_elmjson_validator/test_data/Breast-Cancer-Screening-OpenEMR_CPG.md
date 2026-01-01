# Breast Cancer Screening Logic

## Target Population
Patients must meet **ALL** of the following criteria to be eligible for screening:
1.  **Age:** Patient is **40 years** or older.
2.  **Gender:** Patient gender is **Female**.
3.  **No Recent Screening:** Patient has **0 (zero)** documented "Mammogram" procedures in the past **1 year**.
    * *Note:* Use Value Set `2.16.840.1.113762.1.4.1182.380` to identify Mammogram procedures.

## Recommendation Rule
**IF** the patient is in the Target Population (Female, 40+, No recent mammogram):
**THEN** the system must recommend:
* "Perform mammogram screening"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the Target Population criteria:
**THEN** the system returns no specific error message (Null).