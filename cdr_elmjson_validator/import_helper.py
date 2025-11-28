#!/usr/bin/env python3
"""
ELM Validation Helper Script

Validates all ELM files in CQL Services using Modal/Llama.
Tracks validation status and returns valid libraries for OpenEMR import.

Environment Variables:
    MODAL_VALIDATOR_URL    Required. Your Modal validator endpoint URL
                          Example: https://your-username--elm-validator-validate.modal.run

Usage:
    export MODAL_VALIDATOR_URL='https://your-modal-endpoint.modal.run'
    python3 import_helper.py                    # Validate all unvalidated libraries
    python3 import_helper.py --all              # Re-validate all libraries
    python3 import_helper.py --json             # Output JSON only
    python3 import_helper.py --serve            # Start Flask API server on port 5000
    python3 import_helper.py <library_name>     # Validate specific library
"""

import os
import sys
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime

CQL_SERVICES_PATH = os.environ.get(
    "CQL_SERVICES_PATH",
    str(Path.home() / "AHRQ-CDS-Connect-CQL-SERVICES")
)

MODAL_VALIDATOR_URL = os.environ.get(
    "MODAL_VALIDATOR_URL",
    None
)

def create_flask_app():
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health_check():
        helper_script = Path(CQL_SERVICES_PATH) / "import_helper.py"
        return jsonify({
            "status": "ok",
            "cql_services_path": CQL_SERVICES_PATH,
            "helper_script_exists": helper_script.exists()
        }), 200

    @app.route('/validate', methods=['GET'])
    def validate_libraries():
        try:
            result = validate_all_libraries(force=False, quiet=True)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({
                "error": "Unexpected error",
                "details": str(e)
            }), 500

    @app.route('/library/<library_name>', methods=['GET'])
    def get_library_elm(library_name):
        try:
            libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"
            library_dir = libraries_path / library_name

            if not library_dir.exists():
                return jsonify({"error": f"Library not found: {library_name}"}), 404

            elm_file = find_main_elm_file(library_dir)
            if not elm_file:
                return jsonify({"error": f"No ELM file found for library: {library_name}"}), 404

            with open(elm_file, 'r') as f:
                elm_json = json.load(f)

            return jsonify(elm_json), 200

        except Exception as e:
            return jsonify({
                "error": "Failed to retrieve library",
                "details": str(e)
            }), 500

    return app

def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_validation_status(library_dir, elm_file):
    validation_file = library_dir / ".validation.json"
    current_hash = get_file_hash(elm_file)

    if not validation_file.exists():
        return {"validated": False, "result": None, "hash": current_hash}

    try:
        with open(validation_file, 'r') as f:
            validation_data = json.load(f)

        if validation_data.get("elm_hash") != current_hash:
            return {"validated": False, "result": None, "hash": current_hash}

        return {
            "validated": True,
            "result": validation_data.get("result"),
            "hash": current_hash
        }
    except Exception:
        return {"validated": False, "result": None, "hash": current_hash}

def save_validation_status(library_dir, elm_file, validation_result):
    validation_file = library_dir / ".validation.json"

    validation_data = {
        "elm_hash": get_file_hash(elm_file),
        "validated_at": datetime.now().isoformat(),
        "result": validation_result
    }

    with open(validation_file, 'w') as f:
        json.dump(validation_data, f, indent=2)

def get_all_libraries():
    libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"

    if not libraries_path.exists():
        raise FileNotFoundError(f"Libraries path not found: {libraries_path}")

    libraries = []
    for item in libraries_path.iterdir():
        if item.is_dir():
            if item.name not in ["FHIRHelpers", "Commons", "Conversions"]:
                libraries.append(item.name)

    return libraries

def find_main_elm_file(library_dir):
    for json_file in library_dir.glob("*.json"):
        if any(skip in json_file.name for skip in ["FHIRHelpers", "Commons", "Conversions", "Shared"]):
            continue
        return json_file
    return None

def get_elm_from_local(library_name):
    libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"
    library_dir = libraries_path / library_name

    if not library_dir.exists():
        raise FileNotFoundError(f"Library directory not found: {library_dir}")

    elm_file = find_main_elm_file(library_dir)
    if not elm_file:
        raise FileNotFoundError(f"No main ELM file found in {library_dir}")

    print(f"  Found ELM file: {elm_file.name}")

    with open(elm_file, 'r') as f:
        return json.load(f)

def send_to_modal(elm_json, library_name):
    if not MODAL_VALIDATOR_URL:
        raise ValueError(
            "MODAL_VALIDATOR_URL environment variable is not set. "
            "Please set it to your Modal validator endpoint: "
            "export MODAL_VALIDATOR_URL='https://your-modal-app.modal.run'"
        )

    payload = {
        "elm_json": elm_json,
        "library_name": library_name
    }

    response = requests.post(
        MODAL_VALIDATOR_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=180
    )
    response.raise_for_status()

    return response.json()

def validate_library(library_name, force=False, quiet=False):
    libraries_path = Path(CQL_SERVICES_PATH) / "config" / "libraries"
    library_dir = libraries_path / library_name

    if not library_dir.exists():
        return {
            "library": library_name,
            "valid": False,
            "errors": [f"Library directory not found: {library_dir}"],
            "warnings": [],
            "source": "error"
        }

    elm_file = find_main_elm_file(library_dir)
    if not elm_file:
        return {
            "library": library_name,
            "valid": False,
            "errors": [f"No main ELM file found in {library_dir}"],
            "warnings": [],
            "source": "error"
        }

    if not force:
        status = get_validation_status(library_dir, elm_file)
        if status["validated"]:
            if not quiet:
                print(f"  {library_name}: Already validated (cached)")
            result = status["result"]
            result["library"] = library_name
            return result

    if not quiet:
        print(f"  {library_name}: Validating...")

    try:
        with open(elm_file, 'r') as f:
            elm_json = json.load(f)

        result = send_to_modal(elm_json, library_name)
        result["library"] = library_name

        save_validation_status(library_dir, elm_file, result)

        if not quiet:
            status = "Valid" if result.get("valid") else "Invalid"
            print(f"  {library_name}: {status}")

        return result

    except Exception as e:
        if not quiet:
            print(f"  {library_name}: Error - {e}")
        return {
            "library": library_name,
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "source": "error"
        }

def validate_all_libraries(force=False, quiet=False):
    if not quiet:
        print("=" * 70)
        print("ELM Validation - All Libraries")
        print("=" * 70)
        print(f"CQL Services Path: {CQL_SERVICES_PATH}")
        print(f"Mode: {'Re-validate all' if force else 'Validate new/updated only'}")
        print()

    libraries = get_all_libraries()
    if not quiet:
        print(f"Found {len(libraries)} libraries\n")

    valid_libraries = []
    invalid_libraries = []

    for library_name in libraries:
        result = validate_library(library_name, force=force, quiet=quiet)

        if result.get("valid"):
            valid_libraries.append({
                "name": library_name,
                "source": result.get("source"),
                "warnings": result.get("warnings", [])
            })
        else:
            invalid_libraries.append({
                "name": library_name,
                "errors": result.get("errors", []),
                "source": result.get("source")
            })

    if not quiet:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Valid libraries: {len(valid_libraries)}")
        if valid_libraries:
            for lib in valid_libraries:
                print(f"  - {lib['name']}")

        print(f"\nInvalid libraries: {len(invalid_libraries)}")
        if invalid_libraries:
            for lib in invalid_libraries:
                print(f"  - {lib['name']} ({len(lib['errors'])} errors)")

        print("=" * 70)

    return {
        "valid_libraries": valid_libraries,
        "invalid_libraries": invalid_libraries,
        "total": len(libraries),
        "validated_at": datetime.now().isoformat()
    }

def main():
    force = False
    json_output = False
    serve_mode = False
    library_name = None

    for arg in sys.argv[1:]:
        if arg == "--all":
            force = True
        elif arg == "--json":
            json_output = True
        elif arg == "--serve":
            serve_mode = True
        elif not arg.startswith("--"):
            library_name = arg

    if serve_mode:
        try:
            from flask import Flask
        except ImportError:
            print("Error: Flask is not installed. Install it with: pip3 install flask")
            sys.exit(1)

        helper_script_path = Path(CQL_SERVICES_PATH) / "import_helper.py"
        if not helper_script_path.exists():
            print(f"ERROR: Import helper script not found at {helper_script_path}")
            sys.exit(1)

        print("=" * 70)
        print("CDS Library Validation API")
        print("=" * 70)
        print(f"CQL Services Path: {CQL_SERVICES_PATH}")
        print(f"Import Helper Script: {helper_script_path}")
        print(f"Starting server on http://0.0.0.0:5000")
        print("=" * 70)

        app = create_flask_app()
        app.run(host='0.0.0.0', port=5000, debug=False)
        return

    try:
        if library_name:
            result = validate_library(library_name, force=force, quiet=json_output)

            if json_output:
                print(json.dumps(result, indent=2))

            sys.exit(0 if result.get('valid') else 1)

        else:
            result = validate_all_libraries(force=force, quiet=json_output)

            if json_output:
                print(json.dumps(result, indent=2))

            sys.exit(0 if len(result['valid_libraries']) > 0 else 1)

    except FileNotFoundError as e:
        error = {"error": str(e)}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nError: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        error = {"error": f"Modal request failed: {e}"}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nModal request failed: {e}")
        sys.exit(1)
    except Exception as e:
        error = {"error": str(e)}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
