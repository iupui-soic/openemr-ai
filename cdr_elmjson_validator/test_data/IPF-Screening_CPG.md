# Idiopathic Pulmonary Fibrosis (IPF) Evaluation Clinical Practice Guideline

## Source
Based on ATS/ERS/JRS/ALAT 2022 Clinical Practice Guideline — *Idiopathic Pulmonary Fibrosis (an Update) and Progressive Pulmonary Fibrosis in Adults* (Raghu et al., Am J Respir Crit Care Med 2022;205:e18–e47. DOI: 10.1164/rccm.202202-0399ST)

## Target Population

### Inclusion Criteria
1. **Age**: ≥ 60 years
2. **Condition**: Interstitial Pulmonary Fibrosis [Unspecified Cause] (SNOMED)
3. **Observation**: Interstitial lung disease diagnosis documented

### Exclusion Criteria (patient should NOT meet any of these)
- **Condition**: "There does exist a confirmed active condition with a code from Connective Tissue Disorders (SNOMED)" — value set `2.16.840.1.113762.1.4.1146.3135`, modifiers: Confirmed, Active, Exists
- **OR Medication Statement**: Amiodarone (active)

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age ≥ 60 years
2. AND Has condition of Interstitial Pulmonary Fibrosis (unspecified cause)
3. AND Has interstitial lung disease diagnosis observation

**MeetsExclusionCriteria** = ANY of the following is true:
1. Has as a confirmed active Connective Tissue Disorder. The clinically correct logic would exclude patients *with* CTD, since CTD-ILD is a secondary cause of interstitial lung disease and not IPF.
2. OR Is on active amiodarone therapy

**InPopulation** = MeetsInclusionCriteria AND NOT MeetsExclusionCriteria

## Recommendation Output

When evaluation is needed (InPopulation = true):
- **Recommendation**: "Consider referral to pulmonology for evaluation of suspected idiopathic pulmonary fibrosis. Recommend high-resolution CT (HRCT) of the chest and multidisciplinary discussion (MDD)."
- **Rationale**: "Patients ≥60 years with interstitial lung disease and no identifiable secondary cause (connective tissue disease, drug exposure, environmental exposure) should be evaluated for IPF. Early diagnosis enables initiation of antifibrotic therapy (pirfenidone or nintedanib), which has been shown to slow disease progression."
- **Indicator**: "warning"
- **Source**: ATS/ERS/JRS/ALAT 2022 IPF Clinical Practice Guideline

When evaluation is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null
