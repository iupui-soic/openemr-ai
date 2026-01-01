# Adult Weight Screening Logic

## Target Population
Patients must meet **ALL** of the following criteria to be eligible for screening:
1.  **Age:** Patient is **18 years** or older.
2.  **No Recent Screening:** Patient has **0 (zero)** documented "Body Weight" observations in the past **1 month**.
    * *Note:* Use Value Set `2.16.840.1.113762.1.4.1045.159` to identify Body Weight observations.

## Recommendation Rule
**IF** the patient is in the Target Population (Age 18+ AND No recent weight):
**THEN** the system must recommend:
* "Measure and record patient's body weight"

## Exclusion/Error Rule
**IF** the patient does **NOT** meet the Target Population criteria (e.g., is under 18 OR has a recent weight):
**THEN** the system must return the following error message:
* "Patient does not meet criteria for weight screening reminder. Either patient is under 18 years old or weight has been measured within the last month."