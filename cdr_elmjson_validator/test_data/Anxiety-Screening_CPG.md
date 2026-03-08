# Anxiety Disorder Screening Clinical Practice Guideline

## Source
Based on USPSTF 2023 Recommendation - Anxiety Disorders in Adults Screening

## Target Population

### Inclusion Criteria
1. **Age**: 18 years or older

### No Screening Needed If:
- An anxiety disorder screening (from "Generalized Anxiety Disorder Screening" value set - `2.16.840.1.113883.3.526.3.1542`) has been performed within the past 1 year with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 18 years
2. AND No anxiety disorder screening (GAD-7 or equivalent) performed in the past 1 year (from "Generalized Anxiety Disorder Screening" value set `2.16.840.1.113883.3.526.3.1542`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform anxiety disorder screening using a validated tool (e.g., GAD-7)"
- **Rationale**: "USPSTF recommends screening for anxiety disorders in adults, including pregnant and postpartum persons. Grade B recommendation. Annual screening is recommended using a validated instrument."
- **Indicator**: "warning"
- **Source**: USPSTF 2023 Anxiety Disorders Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null