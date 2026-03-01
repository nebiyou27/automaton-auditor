from __future__ import annotations

from src.nodes import judges


def test_extract_first_json_object_with_leading_and_trailing_noise():
    raw = (
        "I will now answer.\n"
        '{"judge":"Defense","criterion_id":"graph_orchestration","score":"3/5",'
        '"argument":"Evidence meets requirement with minor gaps.",'
        '"cited_evidence":["src/graph.py"]}\n'
        "Additional commentary."
    )
    parsed = judges._extract_first_json_object(raw)
    assert isinstance(parsed, dict)
    assert parsed.get("judge") == "Defense"
    assert parsed.get("score") == "3/5"


def test_normalize_score_accepts_common_non_schema_formats():
    assert judges._normalize_score("3/5") == 3
    assert judges._normalize_score("7 out of 10") == 4
    assert judges._normalize_score("Score: 8/10") == 4
    assert judges._normalize_score("3") == 3


def test_coerce_opinion_from_raw_text_normalizes_and_validates():
    raw = (
        "Some lead-in text\n"
        '{"result":{"judge":"tech_lead","score":"7 out of 10","reasoning":"Architecture is workable and maintainable.","citations":["src/graph.py"]}}'
        "\nTrailing note."
    )
    op = judges._coerce_opinion_from_raw_text(raw, "TechLead", "graph_orchestration")
    assert op is not None
    assert op.judge == "TechLead"
    assert op.criterion_id == "graph_orchestration"
    assert op.score == 4
    assert op.cited_evidence == ["src/graph.py"]
