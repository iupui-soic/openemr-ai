# Colorectal Cancer Average-Risk Screening Clinical Practice Guideline

## Source
Based on USPSTF 2021 Recommendation - Colorectal Cancer Screening (Average Risk)

## Target Population

### Inclusion Criteria
1. **Age**: 45 to 75 years

### No Screening Needed If:
- A colorectal cancer screening test (from "Colorectal Cancer Screening" value set - `2.16.840.1.113883.3.464.1003.198.12.1011`) has been performed within the past 1 year (annual fecal occult blood testing or equivalent)

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 45 years AND Age <= 75 years
2. AND No colorectal cancer screening performed in the past 1 year (from "Colorectal Cancer Screening" value set `2.16.840.1.113883.3.464.1003.198.12.1011`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform colorectal cancer screening using an appropriate modality (FOBT annually, FIT annually, stool DNA every 1-3 years, colonoscopy every 10 years, or CT colonography every 5 years)"
- **Rationale**: "USPSTF recommends screening for colorectal cancer in all adults aged 45-75 years. Grade A recommendation for ages 50-75; Grade B for ages 45-49."
- **Indicator**: "warning"
- **Source**: USPSTF 2021 Colorectal Cancer Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null