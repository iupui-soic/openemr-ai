# Breast Cancer Screening Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Demographics)
1. **Age**: 40 years or older (Screening Age)
2. **Sex**: Female (Gender)

### Screening Criteria

### No Screening Needed If:
- Mammogram procedure (from "Mammogram VS" - `2.16.840.1.113762.1.4.1182.380`) exists within the past 1 year

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Screening Age (age >= 40 years)
2. AND Gender (female)
3. AND Mammogram (no procedure in past 1 year)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform mammogram screening"

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null