# Chlamydia Screening Clinical Practice Guideline

## Target Population

### Inclusion Criteria (In Demographic)
1. **Age**: 16 years or older AND less than 24 years (16-23)
2. **Sex**: Female (per "Female Administrative Sex" value set)

### Sexual Activity Requirement
Patient must be sexually active, defined as having ANY of:
- Conditions indicating sexual activity
- Laboratory tests indicating sexual activity

## Screening Criteria

### No Screening Needed If ANY of:
1. Chlamydia screening results exist within the past 1 year
2. A chlamydia screening procedure request was ordered today or later (pending order)
3. A documented reason for not performing chlamydia screening exists

## Decision Logic

**Need Screening** = ALL of the following must be true:
1. In Demographic (age 16-23 AND female)
2. Sexually Active
3. No Screening (no recent results, no pending orders, no documented refusal)

## Recommendation Output

When screening is needed, generate a Chlamydia Screening Request:
- **Code**: SNOMED 442487003 - "Screening for Chlamydia trachomatis (procedure)"
- **Status**: proposed