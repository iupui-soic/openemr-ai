# Abdominal Aortic Aneurysm (AAA) Screening Clinical Practice Guideline

## Source
Based on USPSTF 2019 Recommendation - Abdominal Aortic Aneurysm Screening

## Target Population

### Inclusion Criteria
1. **Age**: 65 to 75 years
2. **Sex**: Male
3. **Smoking History**: Ever smoked (100+ cigarettes in lifetime) (from "Tobacco User" value set - `2.16.840.1.113883.3.526.3.1170`)

### No Screening Needed If:
- A prior abdominal aortic aneurysm ultrasound screening (from "AAA Screening Ultrasound" value set - `2.16.840.1.113762.1.4.1047.93`) has ever been performed (one-time screening)

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 65 years AND Age <= 75 years
2. AND Patient sex is Male
3. AND Has smoking history (from "Tobacco User" value set `2.16.840.1.113883.3.526.3.1170`)
4. AND No prior AAA ultrasound screening ever (from "AAA Screening Ultrasound" value set `2.16.840.1.113762.1.4.1047.93`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform one-time abdominal ultrasound for abdominal aortic aneurysm screening"
- **Rationale**: "USPSTF recommends one-time screening for abdominal aortic aneurysm (AAA) with ultrasonography in men aged 65-75 years who have ever smoked (100+ cigarettes in lifetime). Grade B recommendation."
- **Indicator**: "warning"
- **Source**: USPSTF 2019 AAA Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null