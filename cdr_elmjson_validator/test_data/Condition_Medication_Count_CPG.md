# Condition and Medication Count

## Purpose
Utility library providing counts of confirmed conditions and medication requests for a patient.

## Definitions

### NumberConfirmedConditions
- **Source**: FHIR Condition resources
- **Filter**: verificationStatus = "confirmed" (code system: `http://terminology.hl7.org/CodeSystem/condition-ver-status`)
- **Output**: Integer count

### NumberMedicationRequests
- **Source**: FHIR MedicationRequest resources
- **Filter**: None (counts all MedicationRequest resources)
- **Output**: Integer count

### HasCount
- **Logic**: (NumberConfirmedConditions + NumberMedicationRequests) > 0
- **Output**: Boolean

### Summary
- **Format**: `{PatientFirstName} has {N} confirmed conditions and {M} MedicationRequests.`
- **Patient Name**: First given name from Patient.name[0].given[0]
- **Output**: String