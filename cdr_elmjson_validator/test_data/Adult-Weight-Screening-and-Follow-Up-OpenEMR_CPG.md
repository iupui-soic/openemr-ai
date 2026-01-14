# Adult Weight Screening and Follow-Up Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Screening Age)
1. **Age**: 18 years or older

### Screening Criteria

### No Screening Needed If:
- Body Weight observation (from "Body Weight VS" - `2.16.840.1.113762.1.4.1045.159`) exists within the past 1 month

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Screening Age (age >= 18 years)
2. AND Body Weight (count of observations in past 1 month = 0)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Measure and record patient's body weight"

When screening is not needed (InPopulation = false):
- **Errors**: "Patient does not meet criteria for weight screening reminder. Either patient is under 18 years old or weight has been measured within the last month."