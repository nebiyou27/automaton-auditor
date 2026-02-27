import json


def load_rubric(path="rubric/week2_rubric.json"):
    # FIXED: C12
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_dimensions_for(rubric, artifact):
    dimensions = rubric.get("dimensions", []) if isinstance(rubric, dict) else []
    if not isinstance(dimensions, list):
        return []

    filtered = [
        d
        for d in dimensions
        if isinstance(d, dict) and d.get("target_artifact") == artifact
    ]

    # Backward compatibility: older rubrics may omit target_artifact.
    if filtered:
        return filtered
    return [d for d in dimensions if isinstance(d, dict)]
