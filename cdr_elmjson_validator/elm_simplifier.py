"""
ELM JSON Simplifier

Converts verbose ELM JSON into human-readable logic summaries.
Works with any standard HL7 ELM JSON format.

This is deterministic code-based extraction, NOT machine learning.
The output is much easier for LLMs to compare against CPG requirements.
"""

import json
from typing import Any


def simplify_elm(elm_json: dict) -> str:
    """
    Convert ELM JSON to a simplified, human-readable format.

    Args:
        elm_json: Parsed ELM JSON dictionary

    Returns:
        Human-readable string summarizing the clinical logic
    """
    library = elm_json.get("library", {})
    lines = []

    # Header
    identifier = library.get("identifier", {})
    lib_name = identifier.get("id", "Unknown")
    lib_version = identifier.get("version", "")
    lines.append(f"# ELM Logic Summary: {lib_name} {lib_version}")
    lines.append("")

    # Value Sets
    value_sets = library.get("valueSets", {}).get("def", [])
    if value_sets:
        lines.append("## Value Sets Referenced")
        for vs in value_sets:
            name = vs.get("name", "Unknown")
            vs_id = vs.get("id", "").split("/")[-1]  # Get OID from URL
            lines.append(f"- {name}: {vs_id}")
        lines.append("")

    # Statements (the actual logic)
    statements = library.get("statements", {}).get("def", [])
    if statements:
        lines.append("## Clinical Logic Definitions")
        lines.append("")

        for stmt in statements:
            name = stmt.get("name", "Unknown")

            # Skip internal Patient definition
            if name == "Patient" and stmt.get("expression", {}).get("type") == "SingletonFrom":
                continue

            # Parse the expression
            expr = stmt.get("expression", {})
            expr_summary = parse_expression(expr, depth=0)

            # Check if it's a function definition
            if stmt.get("type") == "FunctionDef":
                operands = stmt.get("operand", [])
                params = ", ".join(op.get("name", "?") for op in operands)
                lines.append(f"### {name}({params})")
            else:
                lines.append(f"### {name}")

            lines.append(f"{expr_summary}")
            lines.append("")

    return "\n".join(lines)


def parse_expression(expr: dict, depth: int = 0) -> str:
    """
    Recursively parse an ELM expression into human-readable form.

    This handles all standard ELM expression types.
    """
    if not isinstance(expr, dict):
        return str(expr)

    expr_type = expr.get("type", "")
    indent = "  " * depth

    # Literal values
    if expr_type == "Literal":
        value = expr.get("value", "?")
        value_type = expr.get("valueType", "").split("}")[-1]
        return f"{value} ({value_type})"

    # Quantity (time intervals, dosages, etc.)
    if expr_type == "Quantity":
        value = expr.get("value", "?")
        unit = expr.get("unit", "")
        return f"{value} {unit}"

    # Age calculations
    if expr_type == "CalculateAge":
        precision = expr.get("precision", "Year")
        return f"Patient's age in {precision.lower()}s"

    # Comparison operators
    if expr_type in ("GreaterOrEqual", "Greater", "LessOrEqual", "Less", "Equal"):
        operands = expr.get("operand", [])
        if len(operands) >= 2:
            left = parse_expression(operands[0], depth)
            right = parse_expression(operands[1], depth)
            op_map = {
                "GreaterOrEqual": ">=",
                "Greater": ">",
                "LessOrEqual": "<=",
                "Less": "<",
                "Equal": "="
            }
            op = op_map.get(expr_type, "?")
            return f"{left} {op} {right}"

    # Boolean operators
    if expr_type == "And":
        operands = expr.get("operand", [])
        parts = [parse_expression(op, depth) for op in operands]
        return " AND ".join(parts)

    if expr_type == "Or":
        operands = expr.get("operand", [])
        parts = [parse_expression(op, depth) for op in operands]
        return " OR ".join(parts)

    if expr_type == "Not":
        operand = expr.get("operand", {})
        inner = parse_expression(operand, depth)
        return f"NOT ({inner})"

    # Existence checks
    if expr_type == "Exists":
        operand = expr.get("operand", {})
        inner = parse_expression(operand, depth)
        return f"EXISTS ({inner})"

    # Expression references (named definitions)
    if expr_type == "ExpressionRef":
        name = expr.get("name", "?")
        return f"[{name}]"

    # Function calls
    if expr_type == "FunctionRef":
        name = expr.get("name", "?")
        operands = expr.get("operand", [])
        args = [parse_expression(op, depth) for op in operands]
        return f"{name}({', '.join(args)})"

    # Value set references
    if expr_type == "ValueSetRef":
        name = expr.get("name", "?")
        return f'ValueSet "{name}"'

    # Data retrieval (FHIR queries)
    if expr_type == "Retrieve":
        data_type = expr.get("dataType", "").split("}")[-1]
        codes = expr.get("codes", {})
        if codes:
            vs_name = codes.get("name", "?")
            return f'Retrieve {data_type} where type in ValueSet "{vs_name}"'
        return f"Retrieve {data_type}"

    # Conditional expressions
    if expr_type == "If":
        condition = parse_expression(expr.get("condition", {}), depth)
        then_expr = parse_expression(expr.get("then", {}), depth)
        else_expr = parse_expression(expr.get("else", {}), depth)
        return f"IF {condition} THEN {then_expr} ELSE {else_expr}"

    # Null
    if expr_type == "Null":
        return "null"

    # Type casting (often wraps null)
    if expr_type == "As":
        operand = expr.get("operand", {})
        return parse_expression(operand, depth)

    # Interval creation
    if expr_type == "Interval":
        low = expr.get("low", {})
        high = expr.get("high", {})
        low_str = parse_expression(low, depth)
        high_str = parse_expression(high, depth)
        low_closed = "[]"[0] if expr.get("lowClosed", True) else "()"[0]
        high_closed = "[]"[1] if expr.get("highClosed", True) else "()"[1]
        return f"{low_closed}{low_str} to {high_str}{high_closed}"

    # Date/time arithmetic
    if expr_type == "Subtract":
        operands = expr.get("operand", [])
        if len(operands) >= 2:
            left = parse_expression(operands[0], depth)
            right = parse_expression(operands[1], depth)
            return f"{left} - {right}"

    if expr_type == "Now":
        return "Now"

    # Operand references (function parameters)
    if expr_type == "OperandRef":
        name = expr.get("name", "?")
        return f"${name}"

    # Query expressions
    if expr_type == "Query":
        sources = expr.get("source", [])
        where = expr.get("where", {})
        let_clauses = expr.get("let", [])

        parts = []
        for src in sources:
            alias = src.get("alias", "")
            src_expr = parse_expression(src.get("expression", {}), depth)
            parts.append(f"FROM {src_expr} AS {alias}")

        for let in let_clauses:
            let_name = let.get("identifier", "?")
            let_expr = parse_expression(let.get("expression", {}), depth)
            parts.append(f"LET {let_name} = {let_expr}")

        if where:
            where_str = parse_expression(where, depth)
            parts.append(f"WHERE {where_str}")

        return " ".join(parts)

    # Query let references
    if expr_type == "QueryLetRef":
        name = expr.get("name", "?")
        return f"${name}"

    # Overlaps (interval comparison)
    if expr_type == "Overlaps":
        operands = expr.get("operand", [])
        if len(operands) >= 2:
            left = parse_expression(operands[0], depth)
            right = parse_expression(operands[1], depth)
            return f"({left}) OVERLAPS ({right})"

    # Property access
    if expr_type == "Property":
        path = expr.get("path", "?")
        source = expr.get("source", {})
        scope = expr.get("scope", "")
        if scope:
            return f"{scope}.{path}"
        src_str = parse_expression(source, depth)
        return f"{src_str}.{path}"

    # Count
    if expr_type == "Count":
        source = expr.get("source", {})
        return f"COUNT({parse_expression(source, depth)})"

    # Singleton extraction
    if expr_type == "SingletonFrom":
        operand = expr.get("operand", {})
        return parse_expression(operand, depth)

    # Default: return type name for unhandled expressions
    return f"[{expr_type}]"


def extract_key_values(elm_json: dict) -> dict:
    """
    Extract key clinical values from ELM JSON for easy comparison.

    Returns a dictionary of:
    - age_thresholds: List of (comparison, value, unit) tuples
    - time_intervals: List of (value, unit) tuples
    - value_sets: List of (name, oid) tuples
    - conditions: List of condition descriptions
    """
    library = elm_json.get("library", {})

    result = {
        "age_thresholds": [],
        "time_intervals": [],
        "value_sets": [],
        "conditions": [],
    }

    # Extract value sets
    for vs in library.get("valueSets", {}).get("def", []):
        name = vs.get("name", "Unknown")
        vs_id = vs.get("id", "").split("/")[-1]
        result["value_sets"].append((name, vs_id))

    # Recursively find age comparisons and quantities
    def extract_from_expr(expr: dict, context: str = ""):
        if not isinstance(expr, dict):
            return

        expr_type = expr.get("type", "")

        # Age comparisons
        if expr_type in ("GreaterOrEqual", "Greater", "LessOrEqual", "Less", "Equal"):
            operands = expr.get("operand", [])
            if len(operands) >= 2:
                # Check if this is an age comparison
                left = operands[0]
                right = operands[1]

                if left.get("type") == "CalculateAge":
                    precision = left.get("precision", "Year")
                    value = right.get("value", "?")
                    result["age_thresholds"].append({
                        "comparison": expr_type,
                        "value": value,
                        "unit": precision,
                        "context": context
                    })

        # Quantity values (time intervals, etc.)
        if expr_type == "Quantity":
            value = expr.get("value", "?")
            unit = expr.get("unit", "")
            result["time_intervals"].append({
                "value": value,
                "unit": unit,
                "context": context
            })

        # Recurse into nested structures
        for key, val in expr.items():
            if isinstance(val, dict):
                extract_from_expr(val, context)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        extract_from_expr(item, context)

    # Process all statements
    for stmt in library.get("statements", {}).get("def", []):
        name = stmt.get("name", "Unknown")
        expr = stmt.get("expression", {})
        extract_from_expr(expr, context=name)

        # Also check function operands
        for operand in stmt.get("operand", []):
            extract_from_expr(operand, context=name)

    return result


def compare_format(elm_json: dict) -> str:
    """
    Generate a comparison-friendly format for LLM validation.

    This format explicitly lists key values that need to be compared
    against the CPG requirements.
    """
    values = extract_key_values(elm_json)
    library = elm_json.get("library", {})
    identifier = library.get("identifier", {})

    lines = []
    lines.append(f"# ELM Implementation: {identifier.get('id', 'Unknown')}")
    lines.append("")

    lines.append("## Key Values to Verify Against CPG:")
    lines.append("")

    # Age thresholds
    if values["age_thresholds"]:
        lines.append("### Age Thresholds")
        for thresh in values["age_thresholds"]:
            comp_map = {
                "GreaterOrEqual": ">=",
                "Greater": ">",
                "LessOrEqual": "<=",
                "Less": "<",
                "Equal": "="
            }
            comp = comp_map.get(thresh["comparison"], "?")
            lines.append(f"- Age {comp} {thresh['value']} {thresh['unit'].lower()}s (in: {thresh['context']})")
    else:
        lines.append("### Age Thresholds")
        lines.append("- None specified")
    lines.append("")

    # Time intervals
    if values["time_intervals"]:
        lines.append("### Time Intervals (Lookback Periods)")
        for interval in values["time_intervals"]:
            lines.append(f"- {interval['value']} {interval['unit']} (in: {interval['context']})")
    else:
        lines.append("### Time Intervals")
        lines.append("- None specified")
    lines.append("")

    # Value sets
    if values["value_sets"]:
        lines.append("### Value Sets")
        for name, oid in values["value_sets"]:
            lines.append(f"- {name}: {oid}")
    lines.append("")

    # Full logic summary
    lines.append("## Full Logic Summary")
    lines.append("")
    lines.append(simplify_elm(elm_json))

    return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python elm_simplifier.py <elm_file.json>")
        print("\nOutputs a simplified, human-readable version of the ELM logic.")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        elm = json.load(f)

    print("=" * 70)
    print("COMPARISON FORMAT (for LLM validation)")
    print("=" * 70)
    print(compare_format(elm))