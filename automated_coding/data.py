"""Load the CPT-annotated MIMIC-III notes and the 61-code description dictionary.

`load_notes` reads the parquet produced by `make_mdace_cpt.py`. It defaults to a
copy bundled next to the benchmark code (`benchmark/dataset/all.parquet`) so the
benchmark folder is self-contained for copying to a different machine; if that
copy is absent it falls back to the canonical `data/processed/mdace_cpt/` path.

`load_cpt_descriptions` prefers the bundled `benchmark/dataset/codes.json` if it
exists, otherwise walks the raw MDACE Profee JSONs. The 61 codes in
`all.parquet` are all covered with consistent official CPT text in MDACE.
"""
import json
from pathlib import Path
from typing import NamedTuple

import polars as pl


class Note(NamedTuple):
    note_id: str
    text: str
    gold_codes: set[str]


_BENCHMARK_DIR = Path(__file__).resolve().parent
BUNDLED_PARQUET = _BENCHMARK_DIR / "dataset" / "all.parquet"
BUNDLED_CODES_JSON = _BENCHMARK_DIR / "dataset" / "codes.json"

FALLBACK_PARQUET = Path("data/processed/mdace_cpt/all.parquet")
FALLBACK_MDACE_PROFEE_DIR = Path("data/raw/MDace/Profee/ICD-9/1.0")


def default_parquet() -> Path:
    return BUNDLED_PARQUET if BUNDLED_PARQUET.exists() else FALLBACK_PARQUET


DEFAULT_PARQUET = default_parquet()
DEFAULT_MDACE_PROFEE_DIR = FALLBACK_MDACE_PROFEE_DIR


def load_notes(path: Path = DEFAULT_PARQUET) -> list[Note]:
    df = pl.read_parquet(path)
    notes: list[Note] = []
    for row in df.iter_rows(named=True):
        notes.append(
            Note(
                note_id=str(row["note_id"]),
                text=row["text"],
                gold_codes=set(row["procedure_codes"]),
            )
        )
    return notes


def get_label_space(notes: list[Note]) -> list[str]:
    codes: set[str] = set()
    for note in notes:
        codes.update(note.gold_codes)
    return sorted(codes)


def load_cpt_descriptions(
    profee_dir: Path = DEFAULT_MDACE_PROFEE_DIR,
    label_space: list[str] | None = None,
) -> dict[str, str]:
    if BUNDLED_CODES_JSON.exists():
        descriptions = json.loads(BUNDLED_CODES_JSON.read_text(encoding="utf8"))
    else:
        descriptions = {}
        for json_path in sorted(profee_dir.glob("*.json")):
            case = json.loads(json_path.read_text(encoding="utf8"))
            for note in case["notes"]:
                for ann in note["annotations"]:
                    if ann["code_system"] != "CPT":
                        continue
                    code = ann["code"]
                    desc = ann.get("description", "").strip()
                    if not desc:
                        continue
                    if code in descriptions and descriptions[code] != desc:
                        continue
                    descriptions.setdefault(code, desc)

    if label_space is not None:
        missing = [c for c in label_space if c not in descriptions]
        if missing:
            raise ValueError(f"Missing CPT descriptions for: {missing}")
        descriptions = {c: descriptions[c] for c in label_space}
    return descriptions


def write_codes_json(descriptions: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf8"
    )
