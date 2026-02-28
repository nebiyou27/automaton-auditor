import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

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
    dimension_id: str = Field(
        default="",
        description="Rubric dimension id this evidence supports."
    )


class JudicialOpinion(BaseModel):
    """
    One judge's verdict for a single rubric criterion.
    Three opinions (Prosecutor, Defense, TechLead) form a dialectical bench.
    """
    judge: Literal["Prosecutor", "Defense", "TechLead"] = Field(description="Which persona produced this opinion.")
    criterion_id: str = Field(description="Rubric criterion ID being scored.")
    score: int = Field(ge=1, le=5, description="Score from 1 (worst) to 5 (best).")
    argument: str = Field(description="Judge reasoning tied to evidence and rubric logic.")
    cited_evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence locations used to justify this score."
    )


class ToolCall(BaseModel):
    """
    Planned tool execution candidate for the next audit step.
    """
    dimension_id: str = Field(description="Rubric dimension id this tool call targets.")
    tool_name: str = Field(description="Tool identifier to execute.")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments payload."
    )
    why: str = Field(description="Why this tool call should run now.")
    expected_evidence: str = Field(description="What evidence this call is expected to produce.")
    priority: int = Field(
        ge=1,
        le=5,
        description="1 (lowest) to 5 (highest) priority."
    )


class StopDecision(BaseModel):
    """
    Planner stop/go decision based on remaining forensic risk.
    """
    stop: bool = Field(description="Whether the audit can stop safely.")
    reason: str = Field(description="Short rationale for stop/go.")
    remaining_risks: List[str] = Field(
        default_factory=list,
        description="Risks that still require evidence collection."
    )


class ToolRunMetadata(BaseModel):
    """
    Execution metadata for one dispatched tool call.
    """
    dimension_id: str = Field(description="Rubric dimension id for this run.")
    tool_name: str = Field(description="Executed tool name.")
    elapsed_ms: int = Field(ge=0, description="Elapsed runtime in milliseconds.")
    output_size: int = Field(ge=0, description="Approx output size in characters.")


class DimensionReflection(BaseModel):
    """
    Reflective scoring status for one rubric dimension.
    """
    dimension_id: str = Field(description="Rubric dimension id.")
    applicable: bool = Field(description="Whether this dimension is applicable for current inputs.")
    coverage: float = Field(ge=0.0, le=1.0, description="Evidence coverage score in [0,1].")
    confidence: float = Field(ge=0.0, le=1.0, description="Evidence confidence score in [0,1].")
    missing_questions: List[str] = Field(
        default_factory=list,
        description="Unresolved forensic checks/questions."
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
    repo_path: NotRequired[str]

    rubric_path: NotRequired[str]

    evidences: Annotated[
        Dict[str, List[Evidence]],
        operator.ior
    ]

    opinions: Annotated[
        List[JudicialOpinion],
        operator.add
    ]

    planned_tool_calls: NotRequired[List[ToolCall]]
    stop_decision: NotRequired[StopDecision]
    tool_runs: Annotated[
        List[ToolRunMetadata],
        operator.add
    ]
    pdf_index: NotRequired[Dict[str, Any]]
    reflections: NotRequired[List[DimensionReflection]]
    iteration: NotRequired[int]
    max_iters: NotRequired[int]
    tool_budget: NotRequired[int]
    error_type: NotRequired[str]
    error_message: NotRequired[str]
    failed_node: NotRequired[str]

    final_report: str
