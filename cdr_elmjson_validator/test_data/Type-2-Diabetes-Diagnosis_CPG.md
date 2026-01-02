# Type 2 Diabetes Diagnosis Logic

## Target Population
Patients should be flagged for potential Type 2 Diabetes if they meet the following lab criteria:
1.  **Lab Test:** Hemoglobin A1c (HbA1c).
2.  **Threshold:** Result is greater than or equal to **6.5%**.
    * *Note:* Use standard LOINC codes for HbA1c (e.g., `4548-4`, `17856-6`).

## Recommendation Rule
**IF** the patient has an HbA1c result $\ge$ 6.5%:
**THEN** the system must recommend:
* "Diabetes Diagnosis Indicated - Follow up required"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the threshold:
**THEN** the system returns `null`.