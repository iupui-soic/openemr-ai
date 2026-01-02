# Statin Therapy for CVD Prevention (LDL > 190)

## Target Population
Patients must meet **ALL** of the following criteria to be eligible for automatic statin therapy:
1.  **Age:** Patient is between **30 and 85 years** (inclusive).
2.  **High LDL:** Patient has a most recent LDL Cholesterol result greater than or equal to **190 mg/dL**.
    * *Note:* Use Value Set `2.16.840.1.113883.3.526.3.1573`.

## Recommendation Rule
**IF** the patient is in the Target Population (Age 30-85, LDL >= 190):
**THEN** the system must recommend:
* "Start Statin Therapy"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the criteria:
**THEN** the system returns `null`.