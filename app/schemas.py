"""Pydantic request / response schemas."""
from typing import List, Optional

from pydantic import BaseModel, Field

from . import config


class TargetSearchRequest(BaseModel):
    target_name: str = Field(..., min_length=2)
    max_depth: int = Field(default=config.DEFAULT_MAX_DEPTH, ge=1, le=3)


class GraphNode(BaseModel):
    id: str
    label: str
    kind: str = "person"  # "person" | "organization"
    type: Optional[str] = None  # org type when kind == organization


class GraphEdge(BaseModel):
    id: str
    from_: str = Field(..., alias="from")
    to: str
    type: str
    confidence: float
    source_url: Optional[str] = None
    status: str
    method: Optional[str] = None
    evidence: Optional[str] = None
    depth: int = 0

    class Config:
        populate_by_name = True


class GraphStats(BaseModel):
    people_found: int
    organizations_found: int
    edges_found: int
    sources_fetched: int
    nodes_processed_per_depth: List[int] = Field(default_factory=list)


class GraphResponse(BaseModel):
    graph_id: Optional[str] = None  # per-session graph this result belongs to
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: GraphStats
