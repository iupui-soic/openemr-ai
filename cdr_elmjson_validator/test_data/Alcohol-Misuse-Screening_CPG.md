# Alcohol Misuse Screening Clinical Practice Guideline

## Source
Based on USPSTF 2018 Recommendation - Unhealthy Alcohol Use in Adolescents and Adults Screening

## Target Population

### Inclusion Criteria
1. **Age**: 18 years or older

### No Screening Needed If:
- An alcohol misuse screening (from "Alcohol Use Screening" value set - `2.16.840.1.113883.3.600.1542`) has been performed within the past 1 year with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 18 years
2. AND No alcohol misuse screening (AUDIT-C or equivalent) performed in the past 1 year (from "Alcohol Use Screening" value set `2.16.840.1.113883.3.600.1542`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform alcohol misuse screening using a validated tool (e.g., AUDIT-C)"
- **Rationale**: "USPSTF recommends screening for unhealthy alcohol use in primary care settings in adults 18 years or older, including pregnant women, and provide persons engaged in risky or hazardous drinking with brief behavioral counseling interventions. Grade B recommendation."
- **Indicator**: "warning"
- **Source**: USPSTF 2018 Alcohol Misuse Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null