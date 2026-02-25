import operator
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


class Evidence(BaseModel):
    """
    Forensic proof collected by a Detective agent.
    This is factual-only (no scores, no opinions).
    """
    goal: str = Field(description="What the detective attempted to verify or locate.")
    found: bool = Field(description="Whether the targeted artifact or fact was found.")
    content: Optional[str] = Field(
        default=None,
        description="Optional supporting snippet (truncated) or extracted text."
    )
    location: str = Field(description="File path, commit hash, or reference where evidence was found.")
    rationale: str = Field(description="Why this evidence is trustworthy (method used).")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 to 1.0."
    )


class JudicialOpinion(BaseModel):
    """
    One judge's verdict for a single rubric criterion.
    Three opinions (Prosecutor, Defense, TechLead) form a dialectical bench.
    """
    judge: Literal["Prosecutor", "Defense", "TechLead"] = Field(description="Which persona produced this opinion.")
    criterion_id: str = Field(description="Rubric criterion ID being scored (e.g., 'langgraph_architecture').")
    score: int = Field(ge=1, le=5, description="Score from 1 (worst) to 5 (best).")
    argument: str = Field(description="Judge reasoning tied to evidence and rubric logic.")
    cited_evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence locations used to justify this score."
    )


class AgentState(TypedDict):
    """
    Shared state container passed between LangGraph nodes.

    IMPORTANT: Reducers prevent parallel writes from overwriting each other:
      - operator.ior merges dicts (evidence buckets)
      - operator.add concatenates lists (judge opinions)
    """
    repo_url: str
    pdf_path: str

    rubric_path: NotRequired[str]

    evidences: Annotated[
        Dict[str, List[Evidence]],
        operator.ior
    ]

    opinions: Annotated[
        List[JudicialOpinion],
        operator.add
    ]

    final_report: str