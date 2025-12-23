# Clinical Data Summary: Condition and Medication Count

## Purpose
This is a utility CQL library designed to provide a summary count of a patient's confirmed conditions and active medication requests. It is used for clinical dashboards, patient summaries, and care coordination displays.

## Functional Requirements

### 1. Count Confirmed Conditions
- **Data Source**: FHIR Condition resources
- **Filter Criteria**: Only count conditions where `verificationStatus` equals "confirmed"
- **Verification Status Code**:
  - System: `http://terminology.hl7.org/CodeSystem/condition-ver-status`
  - Code: `confirmed`
- **Output**: Integer count of confirmed conditions

### 2. Count Medication Requests
- **Data Source**: FHIR MedicationRequest resources
- **Filter Criteria**: Count all MedicationRequest resources (no status filter)
- **Output**: Integer count of medication requests

### 3. HasCount Flag
- **Logic**: Return `true` if the sum of confirmed conditions and medication requests is greater than 0
- **Formula**: `(NumberConfirmedConditions + NumberMedicationRequests) > 0`
- **Output**: Boolean

### 4. Summary Display
- **Format**: "{PatientFirstName} has {N} confirmed conditions and {M} MedicationRequests."
- **Data Elements**:
  - Patient's first given name (from `Patient.name[0].given[0]`)
  - Count of confirmed conditions
  - Count of medication requests
- **Output**: Human-readable string

## Expected Behavior

| Scenario | Confirmed Conditions | Medication Requests | HasCount | Summary Example |
|----------|---------------------|---------------------|----------|-----------------|
| No data | 0 | 0 | false | "John has 0 confirmed conditions and 0 MedicationRequests." |
| Conditions only | 3 | 0 | true | "John has 3 confirmed conditions and 0 MedicationRequests." |
| Medications only | 0 | 5 | true | "John has 0 confirmed conditions and 5 MedicationRequests." |
| Both present | 2 | 4 | true | "John has 2 confirmed conditions and 4 MedicationRequests." |

## Validation Criteria

### Must Include
1. Retrieval of Condition resources with filter on verificationStatus
2. Retrieval of MedicationRequest resources
3. Count operations on both resource types
4. Boolean comparison for HasCount
5. String concatenation for Summary

### Must Exclude
- Conditions with verificationStatus other than "confirmed" (e.g., provisional, differential, refuted)
- No filtering should be applied to MedicationRequest resources

## FHIR Resource Requirements

### Condition Resource
```json
{
  "resourceType": "Condition",
  "verificationStatus": {
    "coding": [{
      "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
      "code": "confirmed"
    }]
  }
}
```

### MedicationRequest Resource
```json
{
  "resourceType": "MedicationRequest"
}
```

## Clinical Context
This library is typically used in:
- Patient overview dashboards
- Care team coordination views
- Population health summaries
- Clinical decision support context gathering

## Notes
- This is a utility/display library, not a clinical recommendation
- No clinical actions or alerts are triggered by this logic
- The counts are informational only