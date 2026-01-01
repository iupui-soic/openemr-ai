# Colon Cancer Screening Logic

## Target Population
Patients must meet **ALL** of the following criteria to be eligible for screening:
1.  **Age:** Patient is **50 years** or older.
2.  **No Recent Screening:** Patient has **0 (zero)** documented "Colonoscopy" procedures in the past **1 month**.
    * *Note:* Use Value Set `2.16.840.1.113883.3.464.1003.108.12.1020` to identify Colonoscopy procedures.

## Recommendation Rule
**IF** the patient is in the Target Population (Age 50+, No recent colonoscopy):
**THEN** the system must recommend:
* "Perform colon cancer screening (colonoscopy)"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the Target Population criteria:
**THEN** the system returns no specific error message (Null).