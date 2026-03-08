# Cervical Cancer Screening Clinical Practice Guideline

## Source
Based on USPSTF 2018 Recommendation - Cervical Cancer Screening

## Target Population

### Inclusion Criteria
1. **Age**: 21 to 65 years
2. **Sex**: Female

### No Screening Needed If:
- A Pap Test (cervical cytology) (from "Pap Test" value set - `2.16.840.1.113883.3.464.1003.108.12.1017`) has been performed within the past 3 years with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 21 years AND Age <= 65 years
2. AND Patient sex is Female
3. AND No Pap Test (cervical cytology) performed in the past 3 years (from "Pap Test" value set `2.16.840.1.113883.3.464.1003.108.12.1017`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform cervical cytology (Pap smear) for cervical cancer screening"
- **Rationale**: "USPSTF recommends screening for cervical cancer every 3 years with cervical cytology alone in women aged 21-65 years, or every 5 years with high-risk human papillomavirus (hrHPV) testing in combination with cytology in women aged 30-65 years."
- **Indicator**: "warning"
- **Source**: USPSTF 2018 Cervical Cancer Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null