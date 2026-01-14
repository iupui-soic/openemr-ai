# Hypertension Screening Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Minimum Age)
1. **Age**: 18 years or older

### Screening Criteria

### No Screening Needed If:
- Blood pressure screening encounter (from "Encounter to Screen for Blood Pressure VS" - `2.16.840.1.113883.3.600.1920`) exists within the past 1 year (12 months)

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Minimum Age (age >= 18 years)
2. AND Encounter to Screen for Blood Pressure (no encounter in past 1 year)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Screen patient for hypertension"

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null