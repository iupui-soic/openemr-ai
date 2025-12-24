#!/usr/bin/env python3
"""
ELM Validation Runner for CI/CD (v2.0)
Runs ELM JSON validation against specified LLM model using Modal subprocess.
Validates ELM implementations against Clinical Practice Guidelines (CPG).
Compares results against ground truth annotations for accuracy measurement.

Usage:
    python run_validation.py --model llama-3.2-1b --output results.csv
    python run_validation.py --model qwen-2.5-1.5b --output results.csv
    python run_validation.py --all-models --output-dir results/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Available models (must match modal_app.py)
MODELS = [
    "llama-3.2-1b",
    "llama-3.2-3b",
    "qwen-2.5-1.5b",
    "qwen-2.5-3b",
    "phi-3-mini",
    "gemma-3-4b",
    "medgemma-4b",
    "llama-3.1-8b",
    "gemma-3-270m",
]

DEFAULT_MODEL = "llama-3.2-1b"
GROUND_TRUTH_FILE = "ground_truth.json"
TEST_DATA_DIR = "test_data"


def load_ground_truth(data_dir: Path) -> dict:
    """Load ground truth annotations including CPG references."""
    gt_path = data_dir / GROUND_TRUTH_FILE
    if not gt_path.exists():
        print(f"Warning: Ground truth file not found: {gt_path}")
        print("Accuracy metrics will not be available.")
        return {"test_cases": {}}

    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    return gt_data


def load_cpg_content(data_dir: Path, cpg_file: str) -> str:
    """Load Clinical Practice Guideline content from file."""
    if not cpg_file:
        return None

    cpg_path = data_dir / cpg_file
    if not cpg_path.exists():
        print(f"Warning: CPG file not found: {cpg_path}")
        return None

    with open(cpg_path, 'r') as f:
        return f.read()


def get_elm_files(data_dir: Path) -> list:
    """Get all ELM JSON files from directory."""
    elm_files = []
    for json_file in data_dir.glob("*.json"):
        if json_file.name.startswith("."):
            continue
        if json_file.name == GROUND_TRUTH_FILE:
            continue
        elm_files.append(json_file)
    return elm_files


def get_library_name(elm_json: dict, fallback: str = "Unknown") -> str:
    """Extract library name from ELM JSON."""
    library = elm_json.get("library", {})
    identifier = library.get("identifier", {})
    return identifier.get("id", fallback)


def prepare_batch_items(elm_files: list, data_dir: Path, test_cases: dict) -> list:
    """
    Prepare batch items for Modal processing.

    Returns list of dicts with: elm_json, library_name, cpg_content, file_name
    """
    items = []
    for elm_file in elm_files:
        with open(elm_file, 'r') as f:
            elm_json = json.load(f)

        library_name = get_library_name(elm_json, elm_file.stem)
        file_name = elm_file.name
        test_case = test_cases.get(file_name, {})

        # Load CPG if specified
        cpg_file = test_case.get("cpg_file")
        cpg_content = None
        if cpg_file:
            cpg_content = load_cpg_content(data_dir, cpg_file)

        items.append({
            "elm_json": elm_json,
            "library_name": library_name,
            "cpg_content": cpg_content,
            "file_name": file_name,
            "cpg_file": cpg_file
        })

    return items


def validate_batch_with_modal(items: list, model_id: str) -> list:
    """
    Validate multiple ELM files using Modal batch processing.

    This is much more efficient than calling Modal for each file individually
    because the model is loaded once and reused for all files.
    """
    import subprocess

    if not items:
        return []

    script_dir = Path(__file__).parent
    start_time = time.time()

    # Write items to a temp file for Modal to read
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(items, f)
        items_file = f.name

    try:
        # Use modal run with Python to call the batch function directly
        batch_script = f'''
import json
import sys
sys.path.insert(0, "{script_dir}")
from modal_app import get_batch_validator

with open("{items_file}", "r") as f:
    items = json.load(f)

validator = get_batch_validator("{model_id}")
results = validator.remote(items)

# Output results as JSON
print("BATCH_RESULTS_START")
print(json.dumps(results))
print("BATCH_RESULTS_END")
'''

        cmd = ["modal", "run", str(script_dir / "modal_app.py") + "::main"]

        # Actually, we need a different approach - let's add a batch entrypoint to modal_app.py
        # For now, use a simpler approach: run modal_app.py with batch mode
        cmd = [
            "python", "-c", f'''
import json
import modal

# Import the app and batch function
import sys
sys.path.insert(0, "{script_dir}")
from modal_app import app, get_batch_validator

with open("{items_file}", "r") as f:
    items = json.load(f)

with app.run():
    validator = get_batch_validator("{model_id}")
    results = validator.remote(items)

    print("BATCH_RESULTS_START")
    print(json.dumps(results))
    print("BATCH_RESULTS_END")
'''
        ]

        print(f"  Running batch validation for {len(items)} files with {model_id}...")

        result_proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for batch
            cwd=str(script_dir)
        )

        elapsed = time.time() - start_time

        if result_proc.returncode != 0:
            print(f"  Batch Modal command failed (exit {result_proc.returncode})")
            print(f"  STDERR: {result_proc.stderr[:1000]}")
            # Return error results for all items
            return [{
                "file": item["file_name"],
                "library": item["library_name"],
                "model": model_id,
                "valid": False,
                "errors": [f"Modal exit code {result_proc.returncode}: {result_proc.stderr[:200]}"],
                "warnings": [],
                "time_seconds": elapsed / len(items),
                "source": "error",
                "has_cpg": item.get("cpg_content") is not None
            } for item in items]

        # Parse batch results from output
        stdout = result_proc.stdout
        try:
            start_marker = "BATCH_RESULTS_START"
            end_marker = "BATCH_RESULTS_END"
            start_idx = stdout.find(start_marker)
            end_idx = stdout.find(end_marker)

            if start_idx != -1 and end_idx != -1:
                json_str = stdout[start_idx + len(start_marker):end_idx].strip()
                results = json.loads(json_str)

                # Normalize results to match expected format
                normalized = []
                for r in results:
                    normalized.append({
                        "file": r.get("file_name", "unknown"),
                        "library": r.get("library_name", "Unknown"),
                        "model": model_id,
                        "valid": r.get("valid", False),
                        "errors": r.get("errors", []),
                        "warnings": r.get("warnings", []),
                        "time_seconds": r.get("time_seconds", 0),
                        "source": r.get("source", "modal"),
                        "has_cpg": r.get("has_cpg", False),
                        "load_time_seconds": r.get("load_time_seconds", 0),
                        "inference_time_seconds": r.get("inference_time_seconds", 0)
                    })

                print(f"  Batch completed in {elapsed:.2f}s for {len(normalized)} files")
                return normalized
            else:
                raise ValueError("Could not find batch results in output")

        except Exception as e:
            print(f"  Error parsing batch results: {e}")
            print(f"  STDOUT: {stdout[:1000]}")
            return [{
                "file": item["file_name"],
                "library": item["library_name"],
                "model": model_id,
                "valid": False,
                "errors": [f"Error parsing results: {str(e)}"],
                "warnings": [],
                "time_seconds": elapsed / len(items),
                "source": "error",
                "has_cpg": item.get("cpg_content") is not None
            } for item in items]

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  Batch Modal command timed out after {elapsed:.2f}s")
        return [{
            "file": item["file_name"],
            "library": item["library_name"],
            "model": model_id,
            "valid": False,
            "errors": ["Modal batch command timed out after 30 minutes"],
            "warnings": [],
            "time_seconds": elapsed / len(items),
            "source": "error",
            "has_cpg": item.get("cpg_content") is not None
        } for item in items]

    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"  ERROR: {error_msg}")
        traceback.print_exc()
        return [{
            "file": item["file_name"],
            "library": item["library_name"],
            "model": model_id,
            "valid": False,
            "errors": [f"Error: {error_msg}"],
            "warnings": [],
            "time_seconds": elapsed / len(items),
            "source": "error",
            "has_cpg": item.get("cpg_content") is not None
        } for item in items]

    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(items_file)
        except:
            pass


def validate_with_modal(elm_file: Path, model_id: str, cpg_content: str = None, cpg_file_path: Path = None) -> dict:
    """Validate ELM file using Modal via subprocess (single file, for backwards compatibility)."""
    import subprocess

    with open(elm_file, 'r') as f:
        elm_json = json.load(f)

    library_name = get_library_name(elm_json, elm_file.stem)
    start_time = time.time()

    try:
        # Build modal run command
        script_dir = Path(__file__).parent
        cmd = [
            "modal", "run",
            str(script_dir / "modal_app.py"),
            "--elm-file", str(elm_file),
            "--model", model_id
        ]

        # Add CPG file if provided
        if cpg_file_path and cpg_file_path.exists():
            cmd.extend(["--cpg-file", str(cpg_file_path)])

        print(f"  Running: {' '.join(cmd)}")

        # Run modal command and capture output
        result_proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(script_dir)
        )

        elapsed = time.time() - start_time

        # Parse the output to extract results
        stdout = result_proc.stdout
        stderr = result_proc.stderr

        if result_proc.returncode != 0:
            print(f"  Modal command failed (exit {result_proc.returncode})")
            print(f"  STDERR: {stderr[:500]}")
            return {
                "file": elm_file.name,
                "library": library_name,
                "model": model_id,
                "valid": False,
                "errors": [f"Modal exit code {result_proc.returncode}: {stderr[:200]}"],
                "warnings": [],
                "time_seconds": elapsed,
                "source": "error",
                "has_cpg": cpg_content is not None
            }

        # Parse output for validation result
        # Look for "Result: VALID" or "Result: INVALID" in output
        valid = False
        errors = []
        warnings = []

        for line in stdout.split('\n'):
            line_lower = line.lower().strip()
            if 'result: valid' in line_lower:
                valid = True
            elif 'result: invalid' in line_lower:
                valid = False
            elif 'errors:' in line_lower and '[' in line:
                # Try to extract errors list
                try:
                    err_part = line.split('Errors:', 1)[1].strip()
                    if err_part and err_part != '[]':
                        errors.append(err_part)
                except:
                    pass
            elif 'warnings:' in line_lower and '[' in line:
                try:
                    warn_part = line.split('Warnings:', 1)[1].strip()
                    if warn_part and warn_part != '[]':
                        warnings.append(warn_part)
                except:
                    pass

        print(f"  Completed in {elapsed:.2f}s: valid={valid}")

        return {
            "file": elm_file.name,
            "library": library_name,
            "model": model_id,
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "time_seconds": elapsed,
            "source": "modal",
            "has_cpg": cpg_content is not None
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  Modal command timed out after {elapsed:.2f}s")
        return {
            "file": elm_file.name,
            "library": library_name,
            "model": model_id,
            "valid": False,
            "errors": ["Modal command timed out after 10 minutes"],
            "warnings": [],
            "time_seconds": elapsed,
            "source": "error",
            "has_cpg": cpg_content is not None
        }
    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        error_msg = str(e)
        print(f"  ERROR: {error_msg}")
        traceback.print_exc()
        return {
            "file": elm_file.name,
            "library": library_name,
            "model": model_id,
            "valid": False,
            "errors": [f"Error: {error_msg}"],
            "warnings": [],
            "time_seconds": elapsed,
            "source": "error",
            "has_cpg": cpg_content is not None
        }


def compare_with_ground_truth(result: dict, test_case: dict) -> dict:
    """
    Compare LLM result with ground truth.

    Returns metrics:
    - correct: bool - Did the LLM get the validity right?
    - error_match: float - How well did errors match (0-1)?
    - warning_match: float - How well did warnings match (0-1)?
    """
    if not test_case:
        return {
            "has_ground_truth": False,
            "correct": None,
            "error_match": None,
            "warning_match": None
        }

    # Check if validity matches
    expected_valid = test_case.get("valid", True)
    actual_valid = result.get("valid", False)
    correct = expected_valid == actual_valid

    # Check error matching (keyword-based)
    expected_errors = test_case.get("expected_errors", [])
    actual_errors = result.get("errors", [])

    if not expected_errors and not actual_errors:
        error_match = 1.0
    elif not expected_errors:
        error_match = 0.0 if actual_errors else 1.0
    else:
        # Check if expected error keywords appear in actual errors
        matched = 0
        actual_text = " ".join(actual_errors).lower()
        for exp in expected_errors:
            if exp.lower() in actual_text:
                matched += 1
        error_match = matched / len(expected_errors)

    # Check warning matching (keyword-based)
    expected_warnings = test_case.get("expected_warnings", [])
    actual_warnings = result.get("warnings", [])

    if not expected_warnings and not actual_warnings:
        warning_match = 1.0
    elif not expected_warnings:
        warning_match = 0.5  # Neutral if extra warnings
    else:
        matched = 0
        actual_text = " ".join(actual_warnings).lower()
        for exp in expected_warnings:
            if exp.lower() in actual_text:
                matched += 1
        warning_match = matched / len(expected_warnings)

    return {
        "has_ground_truth": True,
        "correct": correct,
        "expected_valid": expected_valid,
        "actual_valid": actual_valid,
        "error_match": error_match,
        "warning_match": warning_match
    }


def run_validation(model_id: str, data_dir: Path, output_file: Path, use_batch: bool = True) -> dict:
    """Run validation for all ELM files with specified model.

    Args:
        model_id: Model identifier
        data_dir: Directory containing ELM JSON files
        output_file: Output CSV file path
        use_batch: If True, process all files in a single Modal invocation (more efficient)
    """
    elm_files = get_elm_files(data_dir)
    ground_truth = load_ground_truth(data_dir)
    test_cases = ground_truth.get("test_cases", {})

    if not elm_files:
        print(f"No ELM JSON files found in {data_dir}")
        return {"results": [], "summary": {}}

    print(f"\n{'='*70}")
    print(f"ELM Validation with {model_id}")
    print(f"{'='*70}")
    print(f"Data Directory: {data_dir}")
    print(f"Files: {len(elm_files)}")
    print(f"Test Cases with Ground Truth: {len(test_cases)}")
    print(f"Mode: {'Batch (efficient)' if use_batch else 'Individual (legacy)'}")
    print()

    results = []
    total_time = 0

    if use_batch and len(elm_files) > 1:
        # BATCH MODE: Process all files in a single Modal invocation
        # This is much more efficient because the model is loaded only once
        print("Preparing batch validation...")
        items = prepare_batch_items(elm_files, data_dir, test_cases)

        batch_results = validate_batch_with_modal(items, model_id)

        # Process results and compare with ground truth
        for result in batch_results:
            file_name = result.get("file", "unknown")
            test_case = test_cases.get(file_name, {})

            # Compare with ground truth
            comparison = compare_with_ground_truth(result, test_case)
            result.update(comparison)

            # Add CPG file info
            for item in items:
                if item["file_name"] == file_name:
                    result["cpg_file"] = item.get("cpg_file")
                    break

            results.append(result)

            status = "VALID" if result.get("valid") else "INVALID"
            time_taken = result.get("time_seconds", 0)
            total_time += time_taken

            if comparison["has_ground_truth"]:
                correct = "✓" if comparison["correct"] else "✗"
                print(f"  {file_name}: {status} ({time_taken:.2f}s) [{correct} vs ground truth]")
            else:
                print(f"  {file_name}: {status} ({time_taken:.2f}s) [no ground truth]")
    else:
        # INDIVIDUAL MODE: Process files one at a time (legacy behavior)
        for i, elm_file in enumerate(elm_files, 1):
            file_name = elm_file.name
            test_case = test_cases.get(file_name, {})

            # Load CPG if specified in ground truth
            cpg_file = test_case.get("cpg_file")
            cpg_file_path = data_dir / cpg_file if cpg_file else None
            cpg_content = load_cpg_content(data_dir, cpg_file) if cpg_file else None

            cpg_indicator = f" [CPG: {cpg_file}]" if cpg_file else ""
            print(f"[{i}/{len(elm_files)}] Validating {file_name}{cpg_indicator}...")

            result = validate_with_modal(elm_file, model_id, cpg_content, cpg_file_path)

            # Compare with ground truth
            comparison = compare_with_ground_truth(result, test_case)
            result.update(comparison)
            result["cpg_file"] = cpg_file

            results.append(result)

            status = "VALID" if result.get("valid") else "INVALID"
            time_taken = result.get("time_seconds", 0)
            total_time += time_taken

            if comparison["has_ground_truth"]:
                correct = "✓" if comparison["correct"] else "✗"
                print(f"  -> {status} ({time_taken:.2f}s) [{correct} vs ground truth]")
            else:
                print(f"  -> {status} ({time_taken:.2f}s) [no ground truth]")

    # Calculate summary statistics
    with_gt = [r for r in results if r.get("has_ground_truth")]
    without_gt = [r for r in results if not r.get("has_ground_truth")]
    with_cpg = [r for r in results if r.get("cpg_file")]

    if with_gt:
        correct_count = sum(1 for r in with_gt if r.get("correct"))
        accuracy = correct_count / len(with_gt)
        avg_error_match = sum(r.get("error_match", 0) for r in with_gt) / len(with_gt)
        avg_warning_match = sum(r.get("warning_match", 0) for r in with_gt) / len(with_gt)
    else:
        accuracy = None
        avg_error_match = None
        avg_warning_match = None
        correct_count = 0

    avg_time = total_time / len(results) if results else 0

    summary = {
        "model": model_id,
        "total_files": len(results),
        "with_ground_truth": len(with_gt),
        "without_ground_truth": len(without_gt),
        "with_cpg": len(with_cpg),
        "correct": correct_count,
        "accuracy": accuracy,
        "avg_error_match": avg_error_match,
        "avg_warning_match": avg_warning_match,
        "total_time_seconds": total_time,
        "avg_time_seconds": avg_time,
        "timestamp": datetime.now().isoformat()
    }

    # Save results to CSV
    import csv
    with open(output_file, 'w', newline='') as f:
        if results:
            fieldnames = [
                "file", "library", "model", "valid", "time_seconds",
                "errors", "warnings", "source", "cpg_file", "has_cpg",
                "has_ground_truth", "correct", "expected_valid",
                "error_match", "warning_match"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                row = r.copy()
                row["errors"] = "; ".join(r.get("errors", []))
                row["warnings"] = "; ".join(r.get("warnings", []))
                writer.writerow(row)

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Files validated against CPG: {len(with_cpg)}")
    if accuracy is not None:
        print(f"Accuracy: {correct_count}/{len(with_gt)} ({accuracy:.1%})")
        print(f"Error Match: {avg_error_match:.1%}")
        print(f"Warning Match: {avg_warning_match:.1%}")
    else:
        print("Accuracy: N/A (no ground truth annotations)")
    print(f"Files without ground truth: {len(without_gt)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time: {avg_time:.2f}s per file")
    print(f"Results saved to: {output_file}")

    return {"results": results, "summary": summary}


def run_all_models(data_dir: Path, output_dir: Path, use_batch: bool = True) -> list:
    """Run validation with all models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_id in MODELS:
        output_file = output_dir / f"results-{model_id}.csv"
        result = run_validation(model_id, data_dir, output_file, use_batch=use_batch)
        all_summaries.append(result["summary"])

    # Save combined summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    return all_summaries


def generate_markdown_summary(summaries: list, output_file: Path = None) -> str:
    """Generate markdown summary for GitHub Actions."""
    lines = []
    lines.append("## ELM Validation Results\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Commit:** `{os.environ.get('GITHUB_SHA', 'local')[:7]}`\n")

    # Check if we have ground truth data
    has_accuracy = any(s.get("accuracy") is not None for s in summaries)
    has_cpg = any(s.get("with_cpg", 0) > 0 for s in summaries)

    if has_cpg:
        lines.append("*Validation performed against Clinical Practice Guidelines (CPG)*\n")

    if has_accuracy:
        lines.append("### Model Comparison (vs Ground Truth)\n")
        lines.append("| Model | Accuracy | Correct/Total | Error Match | Avg Time | Status |")
        lines.append("|-------|----------|---------------|-------------|----------|--------|")
    else:
        lines.append("### Model Comparison\n")
        lines.append("⚠️ **No ground truth annotations found.** Add annotations to `test_data/ground_truth.json` to measure accuracy.\n")
        lines.append("| Model | Valid/Total | Avg Time | Status |")
        lines.append("|-------|-------------|----------|--------|")

    # Sort by accuracy (or by time if no accuracy)
    if has_accuracy:
        sorted_summaries = sorted(
            summaries,
            key=lambda x: x.get("accuracy", 0) if x.get("accuracy") is not None else -1,
            reverse=True
        )
    else:
        sorted_summaries = sorted(summaries, key=lambda x: x.get("avg_time_seconds", 999))

    for s in sorted_summaries:
        accuracy = s.get("accuracy")
        correct = s.get("correct", 0)
        with_gt = s.get("with_ground_truth", 0)
        total = s.get("total_files", 0)
        avg_time = s.get("avg_time_seconds", 0)
        model = s.get("model", "unknown")
        error_match = s.get("avg_error_match")

        # Status icons based on accuracy
        if accuracy is not None:
            if accuracy >= 0.95:
                status = "🥇 Excellent"
            elif accuracy >= 0.90:
                status = "🥈 Very Good"
            elif accuracy >= 0.80:
                status = "🥉 Good"
            elif accuracy >= 0.70:
                status = "🟡 Fair"
            else:
                status = "🔴 Needs Work"

            error_str = f"{error_match:.0%}" if error_match is not None else "N/A"
            lines.append(
                f"| {model} | {accuracy:.1%} | {correct}/{with_gt} | "
                f"{error_str} | {avg_time:.2f}s | {status} |"
            )
        else:
            lines.append(f"| {model} | {total} files | {avg_time:.2f}s | ⚪ No ground truth |")

    if has_accuracy:
        lines.append("\n### Accuracy Comparison\n")
        lines.append("```")

        # Simple ASCII bar chart for accuracy
        max_bar_len = 30
        for s in sorted_summaries:
            model = s.get("model", "unknown")[:15].ljust(15)
            accuracy = s.get("accuracy")
            if accuracy is not None:
                bar_len = int(accuracy * max_bar_len)
                bar = "█" * bar_len + "░" * (max_bar_len - bar_len)
                lines.append(f"{model} {bar} {accuracy:.1%}")
            else:
                lines.append(f"{model} {'░' * max_bar_len} N/A")

        lines.append("```\n")

    lines.append("### Inference Time Comparison\n")
    lines.append("```")

    # Simple ASCII bar chart for timing
    max_bar_len = 30
    max_time = max(s.get("avg_time_seconds", 0) for s in summaries) if summaries else 1
    for s in sorted(summaries, key=lambda x: x.get("avg_time_seconds", 0)):
        model = s.get("model", "unknown")[:15].ljust(15)
        avg_time = s.get("avg_time_seconds", 0)
        bar_len = int((avg_time / max_time) * max_bar_len) if max_time > 0 else 0
        bar = "█" * bar_len + "░" * (max_bar_len - bar_len)
        lines.append(f"{model} {bar} {avg_time:.2f}s")

    lines.append("```\n")

    lines.append("### Legend")
    lines.append("- 🥇 Excellent (≥95% accuracy)")
    lines.append("- 🥈 Very Good (≥90% accuracy)")
    lines.append("- 🥉 Good (≥80% accuracy)")
    lines.append("- 🟡 Fair (≥70% accuracy)")
    lines.append("- 🔴 Needs Work (<70% accuracy)")
    lines.append("- ⚪ No ground truth available")

    if not has_accuracy:
        lines.append("\n### How to Add Ground Truth")
        lines.append("Edit `test_data/ground_truth.json` to add expected validation results:")
        lines.append("```json")
        lines.append('{')
        lines.append('  "test_cases": {')
        lines.append('    "YourELMFile.json": {')
        lines.append('      "valid": true,')
        lines.append('      "cpg_file": "YourCPG.md",')
        lines.append('      "expected_errors": [],')
        lines.append('      "expected_warnings": []')
        lines.append('    }')
        lines.append('  }')
        lines.append('}')
        lines.append("```")

    summary_text = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(summary_text)

    # Write to GitHub step summary if available
    github_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if github_summary:
        with open(github_summary, 'a') as f:
            f.write(summary_text)

    return summary_text


def main():
    parser = argparse.ArgumentParser(description="ELM Validation Runner")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"Model to use for validation (default: {DEFAULT_MODEL})")
    parser.add_argument("--all-models", action="store_true",
                       help="Run validation with all available models")
    parser.add_argument("--data-dir", type=Path,
                       default=Path(__file__).parent / TEST_DATA_DIR,
                       help="Directory containing ELM JSON and CPG files (default: test_data)")
    parser.add_argument("--output", type=Path, default=Path("results.csv"),
                       help="Output CSV file (for single model)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                       help="Output directory (for all models)")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--no-batch", action="store_true",
                       help="Disable batch processing (validate files one at a time)")

    args = parser.parse_args()

    use_batch = not args.no_batch

    if args.list_models:
        print("Available models:")
        for model in MODELS:
            default = " (default)" if model == DEFAULT_MODEL else ""
            print(f"  {model}{default}")
        return

    if args.all_models:
        summaries = run_all_models(args.data_dir, args.output_dir, use_batch=use_batch)
        summary_md = generate_markdown_summary(summaries, args.output_dir / "summary.md")
        print("\n" + summary_md)
    else:
        if args.model not in MODELS:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {', '.join(MODELS)}")
            sys.exit(1)

        result = run_validation(args.model, args.data_dir, args.output, use_batch=use_batch)
        summaries = [result["summary"]]
        summary_md = generate_markdown_summary(summaries)
        print("\n" + summary_md)


if __name__ == "__main__":
    main()