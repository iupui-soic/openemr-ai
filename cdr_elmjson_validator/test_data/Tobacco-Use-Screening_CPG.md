# Tobacco Use Screening and Cessation Counseling Clinical Practice Guideline

## Source
Based on USPSTF 2021 Recommendation - Tobacco Smoking Cessation in Adults, Including Pregnant Persons

## Target Population

### Inclusion Criteria
1. **Age**: 18 years or older

### No Screening Needed If:
- A tobacco use screening (from "Tobacco Use Screening" value set - `2.16.840.1.113883.3.526.3.1278`) has been performed within the past 1 year with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 18 years
2. AND No tobacco use screening performed in the past 1 year (from "Tobacco Use Screening" value set `2.16.840.1.113883.3.526.3.1278`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Screen patient for tobacco use and provide cessation counseling if positive"
- **Rationale**: "USPSTF recommends that clinicians ask all adults about tobacco use, advise them to stop using tobacco, and provide behavioral interventions and U.S. Food and Drug Administration (FDA)-approved pharmacotherapy for cessation. Grade A recommendation."
- **Indicator**: "warning"
- **Source**: USPSTF 2021 Tobacco Cessation Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null