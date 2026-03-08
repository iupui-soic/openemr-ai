# Osteoporosis Screening (DEXA) Clinical Practice Guideline

## Source
Based on USPSTF 2025 Recommendation - Screening for Osteoporosis to Prevent Fractures / CMS249 - Appropriate Use of DXA Scans

## Target Population

### Inclusion Criteria
1. **Age**: 65 years or older
2. **Sex**: Female (from "Female Administrative Sex" value set - `2.16.840.1.113883.3.560.100.2`)

### Exclusion Criteria
- Existing diagnosis of Osteoporosis (from "Osteoporosis" value set - `2.16.840.1.113883.3.464.1003.113.12.1038`)
- Existing diagnosis of Osteopenia (from "Osteopenia" value set - `2.16.840.1.113883.3.464.1003.113.12.1049`)
- History of Osteoporotic Fractures (from "Osteoporotic Fractures" value set - `2.16.840.1.113883.3.464.1003.113.12.1050`)

### No Screening Needed If:
- A DEXA bone density scan (from "DEXA Dual Energy Xray Absorptiometry, Bone Density for Urology Care" value set - `2.16.840.1.113762.1.4.1151.38`) has been performed within the past 2 years with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 65 years
2. AND Patient sex is Female
3. AND No active Osteoporosis diagnosis (from Osteoporosis value set `2.16.840.1.113883.3.464.1003.113.12.1038`)
4. AND No active Osteopenia diagnosis (from Osteopenia value set `2.16.840.1.113883.3.464.1003.113.12.1049`)
5. AND No history of Osteoporotic Fractures (from Osteoporotic Fractures value set `2.16.840.1.113883.3.464.1003.113.12.1050`)
6. AND No DEXA scan performed in the past 2 years (from DEXA value set `2.16.840.1.113762.1.4.1151.38`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Order DEXA bone density scan for osteoporosis screening"
- **Rationale**: "USPSTF recommends screening for osteoporosis with bone measurement testing (DXA) to prevent osteoporotic fractures in women 65 years or older. Grade B recommendation."
- **Indicator**: "warning"
- **Source**: USPSTF 2025 Osteoporosis Screening Recommendation / CMS249

<!-- Note: ELM JSON checks females 65+, excludes existing osteoporosis/osteopenia/fracture diagnoses, and requires no DEXA scan in past 2 years. -->

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null
