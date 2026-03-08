# Falls Prevention Screening Clinical Practice Guideline

## Source
Based on USPSTF 2018 Recommendation - Interventions to Prevent Falls in Community-Dwelling Older Adults

## Target Population

### Inclusion Criteria
1. **Age**: 65 years or older

### No Screening Needed If:
- A falls risk assessment (from "Fall Risk Assessment" value set - `2.16.840.1.113883.3.464.1003.118.12.1028`) has been performed within the past 1 year with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 65 years
2. AND No falls risk assessment performed in the past 1 year (from "Fall Risk Assessment" value set `2.16.840.1.113883.3.464.1003.118.12.1028`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform falls risk assessment and refer to exercise intervention if at increased risk"
- **Rationale**: "USPSTF recommends exercise interventions to prevent falls in community-dwelling adults 65 years or older who are at increased risk for falls. Grade B recommendation. Clinicians should assess fall risk annually."
- **Indicator**: "warning"
- **Source**: USPSTF 2018 Falls Prevention Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null