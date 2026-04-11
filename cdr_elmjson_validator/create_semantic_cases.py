#!/usr/bin/env python3
"""
Create 8 new semantic logic error test cases for the ELM validation benchmark.

Categories (2 cases each):
  1. Missing condition — expression branch deleted from AND chain
  2. Inverted logic — negation added/removed, changing population selection
  3. Wrong nesting — NOT distributed incorrectly across conjunction
  4. Swapped references — value set OID points to wrong clinical concept

Each new invalid case is derived from a valid ELM file, uses the same CPG,
and compiles as valid ELM but implements incorrect clinical logic.
"""

import json
import copy
import os

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def load_json(filename):
    with open(os.path.join(TEST_DATA_DIR, filename)) as f:
        return json.load(f)


def save_json(data, filename):
    path = os.path.join(TEST_DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {filename}")


def find_statement(elm, name):
    """Find a named statement definition in the ELM library."""
    for stmt in elm["library"]["statements"]["def"]:
        if stmt["name"] == name:
            return stmt
    return None


def find_operand_by_ref(operands, ref_name):
    """Find index of an operand that is an ExpressionRef to ref_name,
    or a Not wrapping an ExpressionRef to ref_name."""
    for i, op in enumerate(operands):
        if op.get("type") == "ExpressionRef" and op.get("name") == ref_name:
            return i
        if op.get("type") == "Not":
            inner = op.get("operand", {})
            if isinstance(inner, dict) and inner.get("type") == "ExpressionRef" and inner.get("name") == ref_name:
                return i
    return None


def remove_and_operand(and_node, ref_name):
    """Remove an operand referencing ref_name from a nested AND tree.
    Returns True if found and removed."""
    if and_node.get("type") != "And":
        return False
    operands = and_node.get("operand", [])
    idx = find_operand_by_ref(operands, ref_name)
    if idx is not None:
        operands.pop(idx)
        # If only one operand left, we should collapse, but for ELM
        # validity a single-operand And still works
        return True
    # Recurse into nested Ands
    for op in operands:
        if remove_and_operand(op, ref_name):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 1: Missing Condition (expression branch deleted)
# ═══════════════════════════════════════════════════════════════════════════

def create_aaa_missing_sex_restriction():
    """Remove 'Is Male' from AAA Screening AND chain.
    AAA screening is for males aged 65-75 who have smoked. Removing the
    sex restriction would screen females too, contrary to the CPG."""
    elm = copy.deepcopy(load_json("AAA-Screening.json"))
    stmt = find_statement(elm, "Needs AAA Screening")
    assert stmt, "Statement 'Needs AAA Screening' not found"
    removed = remove_and_operand(stmt["expression"], "Is Male")
    assert removed, "Failed to remove 'Is Male' operand"
    save_json(elm, "AAA-Missing-SexRestriction.json")
    return {
        "valid": False,
        "cpg_file": "AAA-Screening_CPG.md",
        "expected_errors": ["male", "sex", "gender"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: missing condition): Male sex restriction "
                 "removed from AAA screening. Screens females, contrary to USPSTF "
                 "guideline which targets males aged 65-75 with smoking history."
    }


def create_depression_missing_bipolar():
    """Remove NOT(Has Bipolar Diagnosis) from Depression Screening AND chain.
    USPSTF excludes bipolar patients from depression screening. Removing this
    exclusion would incorrectly screen bipolar patients."""
    elm = copy.deepcopy(load_json("Depression_screening.json"))
    stmt = find_statement(elm, "Needs Depression Screening")
    assert stmt, "Statement 'Needs Depression Screening' not found"
    removed = remove_and_operand(stmt["expression"], "Has Bipolar Diagnosis")
    assert removed, "Failed to remove 'Has Bipolar Diagnosis' operand"
    save_json(elm, "Depression-Missing-BipolarExclusion.json")
    return {
        "valid": False,
        "cpg_file": "Depression_screening_CPG.md",
        "expected_errors": ["bipolar", "exclusion", "missing"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: missing condition): Bipolar diagnosis "
                 "exclusion removed from depression screening. Patients with bipolar "
                 "disorder would be incorrectly included, contrary to USPSTF guideline."
    }


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 2: Inverted Logic (negation added/removed)
# ═══════════════════════════════════════════════════════════════════════════

def create_colon_cancer_inverted_procedure():
    """Change Count==0 to Count>0 for colonoscopy check.
    Original: recommend screening when patient has NO recent colonoscopy.
    Inverted: recommend screening only for patients who ALREADY had one."""
    elm = copy.deepcopy(load_json("Colon-Cancer-Screening-OpenEMR.json"))
    stmt = find_statement(elm, "Colonoscopy")
    assert stmt, "Statement 'Colonoscopy' not found"
    expr = stmt["expression"]
    assert expr.get("type") == "Equal", f"Expected Equal, got {expr.get('type')}"
    # Change Equal to Greater (Count > 0 instead of Count == 0)
    expr["type"] = "Greater"
    save_json(elm, "Colon-Cancer-Inverted-Procedure.json")
    return {
        "valid": False,
        "cpg_file": "Colon-Cancer-Screening-OpenEMR_CPG.md",
        "expected_errors": ["inverted", "colonoscopy", "logic", "already"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: inverted logic): Colonoscopy count check "
                 "changed from Equal(0) to Greater(0). CDS recommends screening only "
                 "for patients who ALREADY had a colonoscopy, inverting the intent."
    }


def create_falls_inverted_assessment():
    """Remove NOT from NOT(Has Recent Falls Assessment).
    Original: screen patients who have NOT had a recent assessment.
    Inverted: screen only patients who ALREADY had an assessment."""
    elm = copy.deepcopy(load_json("Falls-Prevention-Screening.json"))
    stmt = find_statement(elm, "Needs Falls Prevention Screening")
    assert stmt, "Statement 'Needs Falls Prevention Screening' not found"
    expr = stmt["expression"]
    assert expr.get("type") == "And"
    # Find the NOT(HasRecent) operand and replace with just the inner ref
    operands = expr["operand"]
    for i, op in enumerate(operands):
        if op.get("type") == "Not":
            inner = op.get("operand", {})
            if isinstance(inner, dict) and inner.get("name") == "Has Recent Falls Assessment":
                # Replace NOT(ref) with just ref
                operands[i] = inner
                break
    save_json(elm, "Falls-Inverted-Assessment.json")
    return {
        "valid": False,
        "cpg_file": "Falls-Prevention-Screening_CPG.md",
        "expected_errors": ["inverted", "assessment", "already", "negation"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: inverted logic): NOT removed from "
                 "NOT(Has Recent Falls Assessment). CDS screens only patients who "
                 "ALREADY had a falls assessment, inverting the screening intent."
    }


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 3: Wrong Nesting (NOT distributed incorrectly)
# ═══════════════════════════════════════════════════════════════════════════

def create_depression_wrong_nesting():
    """Change NOT(Bipolar) AND NOT(Depression) to NOT(Bipolar AND Depression).
    Original: exclude anyone with bipolar OR anyone with depression (De Morgan).
    Wrong: exclude only patients who have BOTH bipolar AND depression."""
    elm = copy.deepcopy(load_json("Depression_screening.json"))
    stmt = find_statement(elm, "Needs Depression Screening")
    assert stmt, "Statement 'Needs Depression Screening' not found"

    # Navigate the nested AND tree to find the two NOT operands
    # Structure: And(And(And(InDemo, NOT(Bipolar)), NOT(Depression)), NOT(Screening))
    # We want to merge NOT(Bipolar) and NOT(Depression) into NOT(Bipolar AND Depression)
    expr = stmt["expression"]

    # Find NOT(Bipolar) and NOT(Depression) and their parent operand lists
    bipolar_ref = {"type": "ExpressionRef", "name": "Has Bipolar Diagnosis"}
    depression_ref = {"type": "ExpressionRef", "name": "Has Depression Diagnosis"}

    # Build the replacement: NOT(Bipolar AND Depression)
    merged_not = {
        "type": "Not",
        "operand": {
            "type": "And",
            "operand": [bipolar_ref, depression_ref]
        }
    }

    # Remove both individual NOT operands and insert the merged one
    # The inner AND has: And(InDemo, NOT(Bipolar))
    inner_and = expr["operand"][0]["operand"][0]  # And(InDemo, NOT(Bipolar))
    # Remove NOT(Bipolar) from inner AND
    inner_and["operand"] = [op for op in inner_and["operand"]
                            if not (op.get("type") == "Not" and
                                    isinstance(op.get("operand"), dict) and
                                    op["operand"].get("name") == "Has Bipolar Diagnosis")]

    # The mid AND has: And(inner_and, NOT(Depression))
    mid_and = expr["operand"][0]  # And(inner_and_result, NOT(Depression))
    # Replace NOT(Depression) with merged NOT(Bipolar AND Depression)
    for i, op in enumerate(mid_and["operand"]):
        if (op.get("type") == "Not" and isinstance(op.get("operand"), dict) and
                op["operand"].get("name") == "Has Depression Diagnosis"):
            mid_and["operand"][i] = merged_not
            break

    save_json(elm, "Depression-WrongNesting.json")
    return {
        "valid": False,
        "cpg_file": "Depression_screening_CPG.md",
        "expected_errors": ["nesting", "bipolar", "depression", "both"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: wrong nesting): NOT(Bipolar) AND "
                 "NOT(Depression) changed to NOT(Bipolar AND Depression). Only "
                 "excludes patients with BOTH conditions instead of EITHER, "
                 "a De Morgan's law violation."
    }


def create_ipf_wrong_exclusion_nesting():
    """Change OR to AND in IPF exclusion criteria.
    Original: exclude if CTD OR Amiodarone (either disqualifies).
    Wrong: exclude only if CTD AND Amiodarone (both required to exclude)."""
    elm = copy.deepcopy(load_json("IPF-Screening.json"))
    stmt = find_statement(elm, "MeetsExclusionCriteria")
    assert stmt, "Statement 'MeetsExclusionCriteria' not found"
    expr = stmt["expression"]
    assert expr.get("type") == "Or", f"Expected Or, got {expr.get('type')}"
    # Change Or to And
    expr["type"] = "And"
    save_json(elm, "IPF-WrongExclusionNesting.json")
    return {
        "valid": False,
        "cpg_file": "IPF-Screening_CPG.md",
        "expected_errors": ["exclusion", "OR", "AND", "connective tissue", "amiodarone"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: wrong nesting): Exclusion criteria changed "
                 "from OR to AND. Patient must have BOTH connective tissue disorder "
                 "AND amiodarone use to be excluded, instead of EITHER. Patients with "
                 "only one contraindication would be incorrectly included."
    }


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY 4: Swapped References (value set OID points to wrong concept)
# ═══════════════════════════════════════════════════════════════════════════

def swap_valueset_oid(elm, old_oid, new_oid):
    """Replace a value set OID throughout the ELM library."""
    count = 0
    for vs in elm.get("library", {}).get("valueSets", {}).get("def", []):
        if vs.get("id") == old_oid:
            vs["id"] = new_oid
            count += 1
    return count


def create_falls_swapped_valueset():
    """Swap Fall Risk Assessment VS with Depression Screening VS.
    CDS would check for depression screenings instead of fall risk assessments,
    as if the developer copy-pasted from the wrong screening rule."""
    elm = copy.deepcopy(load_json("Falls-Prevention-Screening.json"))
    FALLS_VS = "2.16.840.1.113883.3.464.1003.118.12.1028"
    DEPRESSION_VS = "2.16.840.1.113883.3.600.564"  # Adult Depression Screening
    n = swap_valueset_oid(elm, FALLS_VS, DEPRESSION_VS)
    assert n > 0, f"Failed to find Fall Risk Assessment VS ({FALLS_VS})"
    save_json(elm, "Falls-SwappedValueSet.json")
    return {
        "valid": False,
        "cpg_file": "Falls-Prevention-Screening_CPG.md",
        "expected_errors": ["value set", "wrong", "depression", "fall risk"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: swapped reference): Fall Risk Assessment "
                 "value set OID replaced with Adult Depression Screening OID. CDS "
                 "checks for depression screenings instead of fall risk assessments."
    }


def create_alcohol_swapped_valueset():
    """Swap Alcohol Use Screening VS with Tobacco Use Screening VS.
    CDS would check for tobacco screenings instead of alcohol screenings,
    as if the developer copy-pasted from the wrong screening rule."""
    elm = copy.deepcopy(load_json("Alcohol-Misuse-Screening.json"))
    ALCOHOL_VS = "2.16.840.1.113883.3.600.1542"
    TOBACCO_VS = "2.16.840.1.113883.3.526.3.1278"  # Tobacco Use Screening
    n = swap_valueset_oid(elm, ALCOHOL_VS, TOBACCO_VS)
    assert n > 0, f"Failed to find Alcohol Use Screening VS ({ALCOHOL_VS})"
    save_json(elm, "Alcohol-SwappedValueSet.json")
    return {
        "valid": False,
        "cpg_file": "Alcohol-Misuse-Screening_CPG.md",
        "expected_errors": ["value set", "wrong", "tobacco", "alcohol"],
        "expected_warnings": [],
        "notes": "INVALID CASE (semantic: swapped reference): Alcohol Use Screening "
                 "value set OID replaced with Tobacco Use Screening OID. CDS checks "
                 "for tobacco screenings instead of alcohol misuse screenings."
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main: generate all cases and update ground_truth.json
# ═══════════════════════════════════════════════════════════════════════════

CASES = [
    ("AAA-Missing-SexRestriction.json", create_aaa_missing_sex_restriction),
    ("Depression-Missing-BipolarExclusion.json", create_depression_missing_bipolar),
    ("Colon-Cancer-Inverted-Procedure.json", create_colon_cancer_inverted_procedure),
    ("Falls-Inverted-Assessment.json", create_falls_inverted_assessment),
    ("Depression-WrongNesting.json", create_depression_wrong_nesting),
    ("IPF-WrongExclusionNesting.json", create_ipf_wrong_exclusion_nesting),
    ("Falls-SwappedValueSet.json", create_falls_swapped_valueset),
    ("Alcohol-SwappedValueSet.json", create_alcohol_swapped_valueset),
]


def main():
    print("Creating 8 semantic error test cases...\n")

    # Load current ground truth
    gt_path = os.path.join(TEST_DATA_DIR, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)

    new_entries = {}
    for filename, create_fn in CASES:
        print(f"  Creating: {filename}")
        entry = create_fn()
        new_entries[filename] = entry

    # Add to ground truth
    gt["test_cases"].update(new_entries)

    # Save updated ground truth
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"\n  Updated: ground_truth.json")

    # Summary
    n_valid = sum(1 for tc in gt["test_cases"].values() if tc["valid"])
    n_invalid = sum(1 for tc in gt["test_cases"].values() if not tc["valid"])
    n_total = len(gt["test_cases"])
    n_semantic = sum(1 for tc in gt["test_cases"].values()
                     if not tc["valid"] and "semantic" in tc.get("notes", "").lower())
    n_parametric = n_invalid - n_semantic

    print(f"\n  Benchmark: {n_total} cases ({n_valid} valid, {n_invalid} invalid)")
    print(f"  Invalid breakdown: {n_parametric} parametric, {n_semantic} semantic")
    print(f"  Base rate: {n_valid/n_total:.1%}")


if __name__ == "__main__":
    main()
