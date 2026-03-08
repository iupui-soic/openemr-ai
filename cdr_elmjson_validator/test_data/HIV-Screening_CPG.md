# HIV Screening Clinical Practice Guideline

## Source
Based on USPSTF 2019 Recommendation - Human Immunodeficiency Virus (HIV) Infection Screening

## Target Population

### Inclusion Criteria
1. **Age**: 15 to 65 years

### No Screening Needed If:
- An HIV screening test (from "HIV Test" value set - `2.16.840.1.113762.1.4.1056.50`) has been performed within the past 1 year with a result present

## Decision Logic

**MeetsInclusionCriteria** = ALL of the following must be true:
1. Age >= 15 years AND Age <= 65 years
2. AND No HIV screening test performed in the past 1 year (from "HIV Test" value set `2.16.840.1.113762.1.4.1056.50`)

**InPopulation** = MeetsInclusionCriteria

## Recommendation Output

When screening is needed (InPopulation = true):
- **Recommendation**: "Perform HIV screening test"
- **Rationale**: "USPSTF recommends that clinicians screen for HIV infection in adolescents and adults aged 15-65 years. Younger adolescents and older adults who are at increased risk should also be screened. Grade A recommendation."
- **Indicator**: "warning"
- **Source**: USPSTF 2019 HIV Screening Recommendation

When screening is not needed (InPopulation = false):
- **Recommendation**: null
- **Rationale**: null
- **Links**: null
- **Suggestions**: null
- **Errors**: null