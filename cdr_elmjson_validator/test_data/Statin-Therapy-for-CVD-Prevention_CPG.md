# Statin Therapy for CVD Prevention Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Age range)
1. **Age**: 30 years or older AND 85 years or younger (30-85 years inclusive)

### Laboratory Test Criteria (LDL Cholesterol)
Patient must have:
- Most recent LDL Cholesterol result (from "LDL Cholesterol VS" - `2.16.840.1.113883.3.526.3.1573`)
- Result value >= 190 mg/dL

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age range (age >= 30 years AND age <= 85 years)
2. AND LDL Cholesterol (most recent LDL >= 190 mg/dL)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When criteria are met (InPopulation = true):
- **Recommendation**: "Start Statin Therapy"

When criteria are not met (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null