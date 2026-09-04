from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def question_hash(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def build_protected_question_hashes(rows: Iterable[dict], output_path: str | Path) -> set[str]:
    hashes = {question_hash(str(row.get("Question") or "")) for row in rows if str(row.get("Question") or "").strip()}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"algorithm": "sha256-normalized-question-v1", "hashes": sorted(hashes)}, indent=2), encoding="utf-8")
    return hashes


def load_protected_question_hashes(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    hashes = payload.get("hashes")
    if not isinstance(hashes, list) or not all(isinstance(item, str) for item in hashes):
        raise ValueError("protected question hash file is invalid")
    return set(hashes)
