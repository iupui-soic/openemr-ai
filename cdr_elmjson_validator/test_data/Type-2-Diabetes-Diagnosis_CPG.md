# Type 2 Diabetes Diagnosis Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Age Range)
1. **Age**: 18 years or older

### Laboratory Test Criteria (HbA1c Laboratory Test)
Patient must have:
- Most recent HbA1c Laboratory Test result (from "HbA1c Laboratory Test VS" - `2.16.840.1.113883.3.464.1003.198.11.1024`)
- Result value >= 6.5%

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age Range (age >= 18 years)
2. AND HbA1c Laboratory Test (most recent HbA1c >= 6.5%)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When criteria are met (InPopulation = true):
- **Recommendation**: "Diabetes Diagnosis Indicated - Follow up required"

When criteria are not met (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null