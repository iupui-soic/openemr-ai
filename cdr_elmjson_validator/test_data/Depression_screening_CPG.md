# Depression Screening Clinical Practice Guideline

## Source
Based on CMS2 - Preventive Care and Screening: Screening for Depression and Follow-Up Plan (USPSTF)

## Target Population

### Inclusion Criteria
1. **Age**: 12 years or older
2. **Encounter**: Qualifying encounter during measurement period (from "Encounter to Screen for Depression" value set - `2.16.840.1.113883.3.600.1916`)

### Exclusion Criteria
- Active diagnosis of Bipolar Disorder (from "Bipolar Diagnosis" value set - `2.16.840.1.113883.3.600.450`)
- Active diagnosis of Depression already documented (from "Depression Diagnosis" value set - `2.16.840.1.113883.3.600.145`)

### No Screening Needed If:
- A depression screening using a standardized tool (from "Adolescent Depression Screening" value set - `2.16.840.1.113883.3.600.559` OR "Adult Depression Screening" value set - `2.16.840.1.113883.3.600.564`) has been performed within the past 1 year with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 12 years
2. AND No active Bipolar Disorder diagnosis (from Bipolar Diagnosis value set `2.16.840.1.113883.3.600.450`)
3. AND No active Depression Diagnosis (from Depression Diagnosis value set `2.16.840.1.113883.3.600.145`)
4. AND No depression screening performed in the past 1 year (from Adolescent Depression Screening `2.16.840.1.113883.3.600.559` or Adult Depression Screening `2.16.840.1.113883.3.600.564`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform depression screening using a standardized tool (e.g., PHQ-9 for adults, PHQ-A for adolescents)"
- **Rationale**: "USPSTF recommends screening for depression in the general adult population (age 12+), including pregnant and postpartum persons. Screening should be implemented with adequate systems in place to ensure accurate diagnosis, effective treatment, and appropriate follow-up."
- **Indicator**: "warning"
- **Source**: CMS2 / USPSTF Depression Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null
