# Lung Cancer Screening Clinical Practice Guideline

## Source
Based on USPSTF 2021 Recommendation - Lung Cancer Screening

## Target Population

### Inclusion Criteria
1. **Age**: 50 to 80 years
2. **Smoking History**: Active smoker or former smoker (from "Tobacco User" value set - `2.16.840.1.113883.3.526.3.1170`) with a history of at least 20 pack-years who quit within the past 15 years

### No Screening Needed If:
- A low-dose CT (LDCT) scan (from "Low Dose CT Chest" value set - `2.16.840.1.113762.1.4.1047.97`) has been performed within the past 1 year

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 50 years AND Age <= 80 years
2. AND Active smoking history or former smoker (from "Tobacco User" value set `2.16.840.1.113883.3.526.3.1170`)
3. AND No LDCT scan performed in the past 1 year (from "Low Dose CT Chest" value set `2.16.840.1.113762.1.4.1047.97`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform annual low-dose CT (LDCT) scan for lung cancer screening"
- **Rationale**: "USPSTF recommends annual screening for lung cancer with low-dose computed tomography (LDCT) in adults aged 50-80 years who have a 20 pack-year smoking history and currently smoke or have quit within the past 15 years. Grade B recommendation."
- **Indicator**: "warning"
- **Source**: USPSTF 2021 Lung Cancer Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null