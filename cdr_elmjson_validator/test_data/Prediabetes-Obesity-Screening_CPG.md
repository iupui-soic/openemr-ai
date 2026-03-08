# Prediabetes and Type 2 Diabetes Screening (Obesity-Related) Clinical Practice Guideline

## Source
Based on USPSTF 2021 Recommendation - Prediabetes and Type 2 Diabetes Screening in Adults

## Target Population

### Inclusion Criteria
1. **Age**: 35 to 70 years
2. **BMI**: Overweight or obese (Body Mass Index >= 25 kg/m2) (from "BMI Measurement" value set - `2.16.840.1.113883.3.600.1579`)

### No Screening Needed If:
- A blood glucose or HbA1c test (from "Blood Glucose or HbA1c Test" value set - `2.16.840.1.113883.3.464.1003.198.12.1073`) has been performed within the past 3 years with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 35 years AND Age <= 70 years
2. AND Most recent BMI measurement >= 25 kg/m2 (overweight or obese) (from "BMI Measurement" value set `2.16.840.1.113883.3.600.1579`)
3. AND No blood glucose or HbA1c test performed in the past 3 years (from "Blood Glucose or HbA1c Test" value set `2.16.840.1.113883.3.464.1003.198.12.1073`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Screen for prediabetes and type 2 diabetes with fasting blood glucose or HbA1c test"
- **Rationale**: "USPSTF recommends screening for prediabetes and type 2 diabetes in adults aged 35-70 years who have overweight or obesity (BMI >= 25 kg/m2). Clinicians should offer or refer patients with prediabetes to effective preventive interventions. Grade B recommendation. Repeat screening every 3 years if results are normal."
- **Indicator**: "warning"
- **Source**: USPSTF 2021 Prediabetes and Type 2 Diabetes Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null