# Hypertension Screening Logic (Standard)

## Target Population
Patients must meet **ALL** of the following criteria to be eligible for screening:
1.  **Age:** Patient is **18 years** or older. (Standard adult screening age).
2.  **No Recent Screening:** Patient has **0 (zero)** documented "Encounter to Screen for Blood Pressure" in the past **1 year** (12 months).
    * *Note:* Use Value Set `2.16.840.1.113883.3.600.1920`.

## Recommendation Rule
**IF** the patient is in the Target Population:
**THEN** the system must recommend:
* "Screen patient for hypertension"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the Target Population criteria:
**THEN** the system returns `null`.