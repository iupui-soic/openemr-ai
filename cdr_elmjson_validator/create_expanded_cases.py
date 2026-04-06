#!/usr/bin/env python3
"""
Create expanded ELM JSON test cases for the benchmark.

Category A: Parametric variants (wrong numeric values)
Category B: Semantic logic errors (structural/logical changes)

Each new invalid case is derived from an existing valid ELM file with
a specific modification, and uses the same CPG file as its source.
"""

import json
import copy
import os

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def load_json(filename):
    filepath = os.path.join(TEST_DATA_DIR, filename)
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data, filename):
    filepath = os.path.join(TEST_DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {filepath}")


def find_statement(elm, name):
    """Find a named statement definition in the ELM library."""
    for stmt in elm["library"]["statements"]["def"]:
        if stmt["name"] == name:
            return stmt
    return None


# ---------------------------------------------------------------------------
# Helper: recursively find and replace a literal value in an expression tree
# ---------------------------------------------------------------------------
def replace_literal_value(node, old_value, new_value, match_type=None):
    """Recursively find Literal nodes with old_value and replace with new_value.
    If match_type is given, only replace if the Literal's parent comparison is that type.
    Returns number of replacements made.
    """
    if not isinstance(node, dict):
        return 0
    count = 0
    if node.get("type") == "Literal" and str(node.get("value")) == str(old_value):
        if match_type is None:
            node["value"] = str(new_value)
            count += 1
    for key, val in node.items():
        if isinstance(val, dict):
            count += replace_literal_value(val, old_value, new_value, match_type)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    count += replace_literal_value(item, old_value, new_value, match_type)
    return count


def replace_quantity_value(node, old_value, new_value, old_unit=None, new_unit=None):
    """Recursively find Quantity nodes and replace value (and optionally unit)."""
    if not isinstance(node, dict):
        return 0
    count = 0
    if node.get("type") == "Quantity" and node.get("value") == old_value:
        if old_unit is None or node.get("unit") == old_unit:
            node["value"] = new_value
            if new_unit is not None:
                node["unit"] = new_unit
            count += 1
    for key, val in node.items():
        if isinstance(val, dict):
            count += replace_quantity_value(val, old_value, new_value, old_unit, new_unit)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    count += replace_quantity_value(item, old_value, new_value, old_unit, new_unit)
    return count


def replace_age_in_comparison(node, comparison_type, old_age, new_age):
    """Find a comparison (GreaterOrEqual, LessOrEqual, etc.) that contains
    a CalculateAge and a Literal with old_age, and replace the age."""
    if not isinstance(node, dict):
        return 0
    count = 0
    if node.get("type") == comparison_type:
        operands = node.get("operand", [])
        has_calc_age = any(
            isinstance(op, dict) and op.get("type") == "CalculateAge"
            for op in operands
        )
        if has_calc_age:
            for op in operands:
                if isinstance(op, dict) and op.get("type") == "Literal" and str(op.get("value")) == str(old_age):
                    op["value"] = str(new_age)
                    count += 1
    for key, val in node.items():
        if isinstance(val, dict):
            count += replace_age_in_comparison(val, comparison_type, old_age, new_age)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    count += replace_age_in_comparison(item, comparison_type, old_age, new_age)
    return count


def change_node_type(node, from_type, to_type, max_changes=1):
    """Recursively change a node's type field from from_type to to_type."""
    if not isinstance(node, dict):
        return 0
    count = 0
    if node.get("type") == from_type and count < max_changes:
        node["type"] = to_type
        count += 1
        if count >= max_changes:
            return count
    for key, val in node.items():
        if count >= max_changes:
            break
        if isinstance(val, dict):
            count += change_node_type(val, from_type, to_type, max_changes - count)
        elif isinstance(val, list):
            for item in val:
                if count >= max_changes:
                    break
                if isinstance(item, dict):
                    count += change_node_type(item, from_type, to_type, max_changes - count)
    return count


# ============================================================================
# CATEGORY A: Parametric Variants
# ============================================================================

def create_cervical_cancer_wrong_age():
    """Case 1: Change age threshold from 21 to 30."""
    elm = load_json("Cervical-Cancer-Screening.json")
    # The "In Demographic" statement has GreaterOrEqual with value "21"
    stmt = find_statement(elm, "In Demographic")
    n = replace_age_in_comparison(stmt["expression"], "GreaterOrEqual", "21", "30")
    print(f"  Cervical-Cancer-Screening-WrongAge: replaced {n} age value(s) (21->30)")
    save_json(elm, "Cervical-Cancer-Screening-WrongAge.json")


def create_depression_wrong_lookback():
    """Case 2: Change lookback from 1 year to 3 months."""
    elm = load_json("Depression_screening.json")
    # Both "Has Recent Adult Depression Screening" and "Has Recent Adolescent Depression Screening"
    # have Quantity with value=1, unit="years". Change to value=3, unit="months".
    n = replace_quantity_value(elm, 1, 3, old_unit="years", new_unit="months")
    print(f"  Depression-Screening-WrongLookback: replaced {n} lookback value(s) (1 year -> 3 months)")
    save_json(elm, "Depression-Screening-WrongLookback.json")


def create_falls_wrong_age():
    """Case 3: Change age from 65 to 75."""
    elm = load_json("Falls-Prevention-Screening.json")
    stmt = find_statement(elm, "In Demographic")
    n = replace_age_in_comparison(stmt["expression"], "GreaterOrEqual", "65", "75")
    print(f"  Falls-Prevention-WrongAge: replaced {n} age value(s) (65->75)")
    save_json(elm, "Falls-Prevention-WrongAge.json")


def create_alcohol_wrong_age():
    """Case 4: Change age from 18 to 25."""
    elm = load_json("Alcohol-Misuse-Screening.json")
    stmt = find_statement(elm, "In Demographic")
    n = replace_age_in_comparison(stmt["expression"], "GreaterOrEqual", "18", "25")
    print(f"  Alcohol-Misuse-WrongAge: replaced {n} age value(s) (18->25)")
    save_json(elm, "Alcohol-Misuse-WrongAge.json")


def create_osteoporosis_wrong_lookback():
    """Case 5: Change lookback from 2 years to 6 months."""
    elm = load_json("Osteoporosis screening.json")
    n = replace_quantity_value(elm, 2, 6, old_unit="years", new_unit="months")
    print(f"  Osteoporosis-WrongLookback: replaced {n} lookback value(s) (2 years -> 6 months)")
    save_json(elm, "Osteoporosis-WrongLookback.json")


def create_aaa_wrong_age():
    """Case 6: Change lower age from 65 to 55."""
    elm = load_json("AAA-Screening.json")
    stmt = find_statement(elm, "In Demographic")
    n = replace_age_in_comparison(stmt["expression"], "GreaterOrEqual", "65", "55")
    print(f"  AAA-Screening-WrongAge: replaced {n} age value(s) (65->55)")
    save_json(elm, "AAA-Screening-WrongAge.json")


# ============================================================================
# CATEGORY B: Semantic Logic Errors
# ============================================================================

def create_breast_cancer_wrong_operator():
    """Case 7: Change AND to OR in MeetsInclusionCriteria so female OR age>=40 qualifies."""
    elm = load_json("Breast-Cancer-Screening-OpenEMR.json")
    # MeetsInclusionCriteria has nested And nodes.
    # The inner And combines "Screening Age" and "Gender".
    # We change that inner And to Or so it becomes: (age>=40 OR female) AND mammogram_needed
    stmt = find_statement(elm, "MeetsInclusionCriteria")
    expr = stmt["expression"]
    # expr is: And( And(Screening Age, Gender), Mammogram )
    # We want to change the inner And (operand[0]) to Or
    inner = expr["operand"][0]
    if inner.get("type") == "And":
        inner["type"] = "Or"
        print("  Breast-Cancer-WrongOperator: changed inner And -> Or in MeetsInclusionCriteria")
    else:
        print("  WARNING: inner node was not And, was:", inner.get("type"))
    save_json(elm, "Breast-Cancer-WrongOperator.json")


def create_tobacco_missing_exclusion():
    """Case 8: Remove the lookback screening check entirely.
    The original 'Needs Tobacco Use Screening' is:
       And(In Demographic, Not(Has Recent Tobacco Screening))
    We simplify it to just: In Demographic
    So ALL patients 18+ are flagged regardless of prior screening.
    """
    elm = load_json("Tobacco-Use-Screening.json")
    stmt = find_statement(elm, "Needs Tobacco Use Screening")
    # Replace the And expression with just the "In Demographic" reference
    stmt["expression"] = {
        "name": "In Demographic",
        "type": "ExpressionRef"
    }
    print("  Tobacco-Missing-Exclusion: removed screening lookback check from Needs Tobacco Use Screening")
    save_json(elm, "Tobacco-Missing-Exclusion.json")


def create_lung_cancer_missing_subpopulation():
    """Case 9: Remove upper age bound (should be 50-80, make it just >=50).
    The "In Demographic" is And(age>=50, age<=80). We change it to just age>=50.
    """
    elm = load_json("Lung-Cancer-Screening.json")
    stmt = find_statement(elm, "In Demographic")
    # Currently: And(GreaterOrEqual(age,50), LessOrEqual(age,80))
    # Replace with just the GreaterOrEqual part
    and_expr = stmt["expression"]
    if and_expr.get("type") == "And":
        # The first operand is GreaterOrEqual (age >= 50)
        ge_operand = and_expr["operand"][0]
        if ge_operand.get("type") == "GreaterOrEqual":
            stmt["expression"] = ge_operand
            print("  Lung-Cancer-Missing-SubPopulation: removed upper age bound (<=80) from In Demographic")
        else:
            print("  WARNING: first operand was not GreaterOrEqual, was:", ge_operand.get("type"))
    else:
        print("  WARNING: In Demographic was not And, was:", and_expr.get("type"))
    save_json(elm, "Lung-Cancer-Missing-SubPopulation.json")


# ============================================================================
# Update ground_truth.json
# ============================================================================

def update_ground_truth():
    gt = load_json("ground_truth.json")

    new_cases = {
        "Cervical-Cancer-Screening-WrongAge.json": {
            "valid": False,
            "cpg_file": "Cervical-Cancer-Screening_CPG.md",
            "expected_errors": [
                "wrong age",
                "age 30",
                "lower bound mismatch",
                "should be 21"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (parametric): Lower age threshold changed from 21 to 30 — misses women aged 21-29 who should be screened per USPSTF 2018."
        },
        "Depression-Screening-WrongLookback.json": {
            "valid": False,
            "cpg_file": "Depression_screening_CPG.md",
            "expected_errors": [
                "frequency mismatch",
                "3 months",
                "lookback too short",
                "should be 1 year"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (parametric): Lookback period changed from 1 year to 3 months — would trigger excessive re-screening, not aligned with annual USPSTF recommendation."
        },
        "Falls-Prevention-WrongAge.json": {
            "valid": False,
            "cpg_file": "Falls-Prevention-Screening_CPG.md",
            "expected_errors": [
                "wrong age",
                "age 75",
                "lower bound mismatch",
                "should be 65"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (parametric): Age threshold changed from 65 to 75 — misses adults aged 65-74 who should be screened per USPSTF 2018."
        },
        "Alcohol-Misuse-WrongAge.json": {
            "valid": False,
            "cpg_file": "Alcohol-Misuse-Screening_CPG.md",
            "expected_errors": [
                "wrong age",
                "age 25",
                "lower bound mismatch",
                "should be 18"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (parametric): Age threshold changed from 18 to 25 — misses adults aged 18-24 who should be screened per USPSTF 2018."
        },
        "Osteoporosis-WrongLookback.json": {
            "valid": False,
            "cpg_file": "Osteoporosis-Screening_CPG.md",
            "expected_errors": [
                "frequency mismatch",
                "6 months",
                "lookback too short",
                "should be 2 years"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (parametric): Lookback period changed from 2 years to 6 months — would trigger excessive re-screening with DEXA, not aligned with USPSTF/CMS249 recommendation."
        },
        "AAA-Screening-WrongAge.json": {
            "valid": False,
            "cpg_file": "AAA-Screening_CPG.md",
            "expected_errors": [
                "wrong age",
                "age 55",
                "lower bound mismatch",
                "should be 65"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (parametric): Lower age changed from 65 to 55 — screens men 55-64 who are outside the USPSTF 2019 recommended range of 65-75."
        },
        "Breast-Cancer-WrongOperator.json": {
            "valid": False,
            "cpg_file": "Breast-Cancer-Screening-OpenEMR_CPG.md",
            "expected_errors": [
                "wrong operator",
                "OR instead of AND",
                "logic error",
                "inclusion criteria"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (semantic): Inner AND changed to OR in MeetsInclusionCriteria — patient qualifies if female OR age>=40 (instead of AND), incorrectly screening males or underage patients."
        },
        "Tobacco-Missing-Exclusion.json": {
            "valid": False,
            "cpg_file": "Tobacco-Use-Screening_CPG.md",
            "expected_errors": [
                "missing exclusion",
                "no lookback check",
                "missing screening history",
                "over-screening"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (semantic): Removed the lookback period check for prior tobacco screening — ALL adults 18+ are flagged regardless of whether they were already screened in the past year."
        },
        "Lung-Cancer-Missing-SubPopulation.json": {
            "valid": False,
            "cpg_file": "Lung-Cancer-Screening_CPG.md",
            "expected_errors": [
                "missing upper age limit",
                "no age cap",
                "age 80",
                "over-inclusive population"
            ],
            "expected_warnings": [],
            "notes": "INVALID CASE (semantic): Removed upper age bound of 80 from demographic check — screens all smokers age 50+ with no upper limit, contrary to USPSTF 2021 range of 50-80."
        }
    }

    gt["test_cases"].update(new_cases)
    save_json(gt, "ground_truth.json")
    print(f"\n  ground_truth.json updated with {len(new_cases)} new test cases.")
    print(f"  Total test cases: {len(gt['test_cases'])}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Creating expanded ELM JSON test cases")
    print("=" * 60)

    print("\n--- CATEGORY A: Parametric Variants ---")
    create_cervical_cancer_wrong_age()
    create_depression_wrong_lookback()
    create_falls_wrong_age()
    create_alcohol_wrong_age()
    create_osteoporosis_wrong_lookback()
    create_aaa_wrong_age()

    print("\n--- CATEGORY B: Semantic Logic Errors ---")
    create_breast_cancer_wrong_operator()
    create_tobacco_missing_exclusion()
    create_lung_cancer_missing_subpopulation()

    print("\n--- Updating ground_truth.json ---")
    update_ground_truth()

    print("\nDone!")


if __name__ == "__main__":
    main()
