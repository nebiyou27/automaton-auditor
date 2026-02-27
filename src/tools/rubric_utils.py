import json


def load_rubric(path="rubric/week2_rubric.json"):
    # FIXED: C12
    return json.load(open(path, encoding="utf-8"))


def get_dimensions_for(rubric, artifact):
    return [d for d in rubric["dimensions"] if d["target_artifact"] == artifact]
