# Colon Cancer Screening Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Required Age)
1. **Age**: 50 years or older

### Screening Criteria

### No Screening Needed If:
- Colonoscopy procedure (from "Colonoscopy VS" - `2.16.840.1.113883.3.464.1003.108.12.1020`) exists within the past 1 month

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Required Age (age >= 50 years)
2. AND Colonoscopy (count of procedures in past 1 month = 0)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform colon cancer screening (colonoscopy)"

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null