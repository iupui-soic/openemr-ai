"""
Test runner for validate_code() against the hand-verified answer key in
automated_coding/validator_testset/cases.json.

Usage:
    python3 -m automated_coding.run_validator_tests
"""
import json
from pathlib import Path

import modal

MODAL_APP_NAME = "automated-cpt-icd-coding"
MODAL_CLASS_NAME = "Gemma4Coder"


def main():
    cases_path = Path(__file__).parent / "validator_testset" / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf8"))

    Gemma4Coder = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    coder = Gemma4Coder()

    print(f"Warming up container...")
    coder.wakeup.remote()
    print(f"Running {len(cases)} validator test cases...\n")

    correct = 0
    results = []
    for case in cases:
        result = coder.validate_code.remote(
            note_text=case["note"], code=case["code"], description=case["description"]
        )
        expected = case["expected_supported"]
        actual = result["supported"]
        passed = expected == actual
        correct += passed
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['case_id']} ({case['code']}): expected={expected} actual={actual}")
        if not passed:
            print(f"       reason given: {result.get('reason', '')}")
        results.append({**case, "actual_supported": actual, "passed": passed, "model_reason": result.get("reason", "")})

    print(f"\n{correct}/{len(cases)} correct ({100*correct/len(cases):.0f}%)")

    out_path = Path(__file__).parent / "validator_testset" / "last_run_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf8")
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
