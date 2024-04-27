"""Models for the knowledge graph."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class Node(BaseModel):
    """Model for a node."""

    node_id: int
    name: str
    description: str


class Link(BaseModel):
    """Model for a link between two nodes."""

    link_id: int
    name: str
    node_id_from: int
    node_id_to: int
    description: str


class KnowledgeGraph(BaseModel):
    """Generic representation of a knowledge graph."""

    nodes: List[Node]
    links: List[Link]
    name: str
