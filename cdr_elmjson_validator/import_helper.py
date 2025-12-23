#!/usr/bin/env python3
"""
ELM JSON Validation Helper

Validates ELM JSON files from local folder or external REST service using Modal/Llama.

Environment Variables:
    MODAL_VALIDATOR_URL    Required. Your Modal validator endpoint URL
                          Example: https://your-username--elm-validator-validate.modal.run
    ELM_SERVICE_URL        Optional. External REST service URL to fetch ELM JSON
                          Example: https://cql-services.example.com/api/library

Usage:
    python3 import_helper.py                    # Validate all local ELM files
    python3 import_helper.py --all              # Re-validate all (ignore cache)
    python3 import_helper.py --json             # Output JSON only
    python3 import_helper.py <library_name>     # Validate specific library
    python3 import_helper.py --from-service     # Fetch and validate from external service
"""

import os
import sys
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime

# Local ELM JSON files directory (test_data subfolder)
ELM_DATA_DIR = Path(__file__).parent / "test_data"

MODAL_VALIDATOR_URL = os.environ.get("MODAL_VALIDATOR_URL", None)
ELM_SERVICE_URL = os.environ.get("ELM_SERVICE_URL", None)


def get_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_json_hash(elm_json):
    """Calculate MD5 hash of JSON content."""
    return hashlib.md5(json.dumps(elm_json, sort_keys=True).encode()).hexdigest()


def get_all_elm_files():
    """Get all ELM JSON files from the local data directory."""
    elm_files = []
    for json_file in ELM_DATA_DIR.glob("*.json"):
        if json_file.name.startswith("."):
            continue
        elm_files.append(json_file)
    return elm_files


def find_elm_file(library_name):
    """Find an ELM file by library name (partial match supported)."""
    for json_file in ELM_DATA_DIR.glob("*.json"):
        if json_file.name.startswith("."):
            continue
        if library_name in json_file.stem or json_file.stem == library_name:
            return json_file
    return None


def get_library_name_from_json(elm_json):
    """Extract library name from ELM JSON content."""
    library = elm_json.get("library", {})
    identifier = library.get("identifier", {})
    return identifier.get("id", "Unknown")


def load_elm_from_file(file_path):
    """Load ELM JSON from a local file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def fetch_elm_from_service(library_name):
    """Fetch ELM JSON from an external REST service."""
    if not ELM_SERVICE_URL:
        raise ValueError(
            "ELM_SERVICE_URL environment variable is not set. "
            "Set it to fetch ELM from external service."
        )

    url = f"{ELM_SERVICE_URL.rstrip('/')}/{library_name}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_validation_cache(cache_key):
    """Check if validation result is cached."""
    cache_file = ELM_DATA_DIR / f".validation_{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_validation_cache(cache_key, elm_hash, validation_result):
    """Save validation result to cache."""
    cache_file = ELM_DATA_DIR / f".validation_{cache_key}.json"

    cache_data = {
        "elm_hash": elm_hash,
        "validated_at": datetime.now().isoformat(),
        "result": validation_result
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def send_to_modal(elm_json, library_name):
    """Send ELM JSON to Modal validator service."""
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


def validate_elm_json(elm_json, source_name=None, force=False, quiet=False):
    """
    Validate ELM JSON content.

    Args:
        elm_json: The ELM JSON dictionary to validate
        source_name: Optional name for caching (file name or library name)
        force: If True, skip cache and re-validate
        quiet: If True, suppress console output

    Returns:
        Validation result dictionary
    """
    library_name = get_library_name_from_json(elm_json)
    cache_key = source_name or library_name
    elm_hash = get_json_hash(elm_json)

    # Check cache
    if not force:
        cached = get_validation_cache(cache_key)
        if cached and cached.get("elm_hash") == elm_hash:
            if not quiet:
                print(f"  {library_name}: Already validated (cached)")
            result = cached.get("result", {})
            result["library"] = library_name
            result["source"] = "cache"
            return result

    if not quiet:
        print(f"  {library_name}: Validating...")

    try:
        result = send_to_modal(elm_json, library_name)
        result["library"] = library_name

        save_validation_cache(cache_key, elm_hash, result)

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


def validate_file(file_path, force=False, quiet=False):
    """Validate an ELM JSON file."""
    elm_json = load_elm_from_file(file_path)
    return validate_elm_json(elm_json, source_name=file_path.stem, force=force, quiet=quiet)


def validate_from_service(library_name, force=False, quiet=False):
    """Fetch ELM JSON from external service and validate."""
    if not quiet:
        print(f"  Fetching {library_name} from service...")

    elm_json = fetch_elm_from_service(library_name)
    return validate_elm_json(elm_json, source_name=library_name, force=force, quiet=quiet)


def validate_all_local(force=False, quiet=False):
    """Validate all local ELM JSON files."""
    if not quiet:
        print("=" * 70)
        print("ELM JSON Validation")
        print("=" * 70)
        print(f"Data Directory: {ELM_DATA_DIR}")
        print(f"Mode: {'Re-validate all' if force else 'Validate new/updated only'}")
        print()

    elm_files = get_all_elm_files()
    if not quiet:
        print(f"Found {len(elm_files)} ELM files\n")

    if not elm_files:
        return {
            "valid_libraries": [],
            "invalid_libraries": [],
            "total": 0,
            "validated_at": datetime.now().isoformat()
        }

    valid_libraries = []
    invalid_libraries = []

    for elm_file in elm_files:
        result = validate_file(elm_file, force=force, quiet=quiet)

        entry = {
            "name": result.get("library"),
            "file": elm_file.name,
            "source": result.get("source")
        }

        if result.get("valid"):
            entry["warnings"] = result.get("warnings", [])
            valid_libraries.append(entry)
        else:
            entry["errors"] = result.get("errors", [])
            invalid_libraries.append(entry)

    if not quiet:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Valid: {len(valid_libraries)}")
        for lib in valid_libraries:
            print(f"  - {lib['name']} ({lib['file']})")

        print(f"\nInvalid: {len(invalid_libraries)}")
        for lib in invalid_libraries:
            print(f"  - {lib['name']} ({len(lib['errors'])} errors)")

        print("=" * 70)

    return {
        "valid_libraries": valid_libraries,
        "invalid_libraries": invalid_libraries,
        "total": len(elm_files),
        "validated_at": datetime.now().isoformat()
    }


def main():
    force = False
    json_output = False
    from_service = False
    library_name = None

    for arg in sys.argv[1:]:
        if arg == "--all":
            force = True
        elif arg == "--json":
            json_output = True
        elif arg == "--from-service":
            from_service = True
        elif not arg.startswith("--"):
            library_name = arg

    try:
        if library_name:
            # Validate specific library
            if from_service:
                result = validate_from_service(library_name, force=force, quiet=json_output)
            else:
                elm_file = find_elm_file(library_name)
                if not elm_file:
                    result = {
                        "library": library_name,
                        "valid": False,
                        "errors": [f"ELM file not found: {library_name}"],
                        "warnings": [],
                        "source": "error"
                    }
                else:
                    result = validate_file(elm_file, force=force, quiet=json_output)

            if json_output:
                print(json.dumps(result, indent=2))

            sys.exit(0 if result.get('valid') else 1)

        else:
            # Validate all local files
            result = validate_all_local(force=force, quiet=json_output)

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
        error = {"error": f"Request failed: {e}"}
        if json_output:
            print(json.dumps(error))
        else:
            print(f"\nRequest failed: {e}")
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