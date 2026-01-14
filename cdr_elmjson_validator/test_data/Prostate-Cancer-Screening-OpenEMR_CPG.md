# Prostate Cancer Screening Clinical Practice Guideline

## Target Population

### Inclusion Criteria (Demographics)
1. **Age**: 50 years or older (Screening Age)
2. **Sex**: Male (Patient Gender)

### Screening Criteria

### Procedures Checked (generic_procedure_valuesets)
The system checks for ANY of the following procedures:
- Prostate Specific Antigen Test (from "Prostate Specific Antigen Test VS" - `2.16.840.1.113883.3.526.2.215`)
- Digital Rectal Examination (from "Digital Rectal Examination VS" - `2.16.840.1.113883.3.7643.3.1042`)
- Prostate biopsy (from "Prostate biopsy VS" - `2.16.840.1.113762.1.4.1078.1248`)

### No Screening Needed If:
- Any of the above procedures exist within the past 1 month

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Screening Age (age >= 50 years)
2. AND Patient Gender (male)
3. AND Assessment (no procedure from generic_procedure_valuesets in past 1 month)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform prostate cancer screening (PSA test, DRE, or biopsy as appropriate)"

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null