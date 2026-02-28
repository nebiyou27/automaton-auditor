from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rubric_ids import CANONICAL_DIMENSION_IDS, LEGACY_ID_ALIASES

AUDIT_DIR = ROOT / "audit"
REQUIRED_FOLDERS = (
    "report_onself_generated",
    "report_onpeer_generated",
    "report_bypeer_received",
)


def _extract_dimension_ids(text: str) -> list[str]:
    ids: list[str] = []
    for pat in (r"^###\s+Dimension:\s+([a-z0-9_]+)\s*$", r"^###\s+Criterion:\s+([a-z0-9_]+)\s*$"):
        ids.extend(re.findall(pat, text, flags=re.MULTILINE))
    return ids


def main() -> int:
    errors: list[str] = []

    for folder in REQUIRED_FOLDERS:
        fpath = AUDIT_DIR / folder
        if not fpath.exists():
            errors.append(f"Missing folder: audit/{folder}")
            continue
        reports = sorted(fpath.glob("*.md"))
        if not reports:
            errors.append(f"No markdown reports found in audit/{folder}")
            continue

        # Validate only the latest report in each folder.
        latest = reports[-1]
        text = latest.read_text(encoding="utf-8", errors="ignore")
        dim_ids = _extract_dimension_ids(text)
        dim_set = set(dim_ids)

        legacy_hits = sorted({legacy for legacy in LEGACY_ID_ALIASES if legacy in text})
        if legacy_hits:
            errors.append(f"{latest}: contains legacy ids: {', '.join(legacy_hits)}")

        missing = [d for d in CANONICAL_DIMENSION_IDS if d not in dim_set]
        if missing:
            errors.append(f"{latest}: missing canonical dimension headings: {', '.join(missing)}")

        extra = [d for d in dim_set if d not in set(CANONICAL_DIMENSION_IDS)]
        if extra:
            errors.append(f"{latest}: contains non-canonical dimension headings: {', '.join(sorted(extra))}")

    if errors:
        print("REPORT VALIDATION FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("REPORT VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
