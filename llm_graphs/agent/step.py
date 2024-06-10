from __future__ import annotations

from typing import Optional

import instructor
from instructor import Instructor
from openai import OpenAI

from llm_graphs.agent.messages import (
    system_graph_creator,
    user_generate_graph,
    user_improve_from_feedback,
    user_rate_graph,
)
from llm_graphs.models import (
    Feedback,
    KnowledgeGraph,
)


def _get_client(client: Optional[Instructor]) -> Instructor:
    if client is None:
        return instructor.from_openai(OpenAI())
    return client


def generate_seed_graph(
    model: str,
    goal_str: str,
    meaning_str: str,
    client: Optional[Instructor] = None,
) -> KnowledgeGraph:
    _client = _get_client(client)
    return _client.chat.completions.create(
        model=model,
        response_model=KnowledgeGraph,
        messages=[system_graph_creator(), user_generate_graph(goal_str, meaning_str)],
    )


def rate_graph(
    model: str,
    goal_str: str,
    meaning_str: str,
    knowledge_graph: KnowledgeGraph,
    client: Optional[Instructor] = None,
) -> Feedback:
    _client = _get_client(client)
    return _client.chat.completions.create(
        model=model,
        response_model=Feedback,
        messages=[system_graph_creator(), user_rate_graph(goal_str, meaning_str, knowledge_graph)],
    )


def new_graph_from_feedback(
    model: str,
    goal_str: str,
    meaning_str: str,
    last_knowledge_graph: KnowledgeGraph,
    last_feedback: Feedback,
    client: Optional[Instructor] = None,
) -> KnowledgeGraph:
    _client = _get_client(client)
    return _client.chat.completions.create(
        model=model,
        response_model=KnowledgeGraph,
        messages=[
            system_graph_creator(),
            user_improve_from_feedback(goal_str, meaning_str, last_knowledge_graph, last_feedback),
        ],
    )
