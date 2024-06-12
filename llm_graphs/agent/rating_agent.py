"""Module to iteratively generate and rate knowledge graphs for a given book."""
# pylint: disable=BLE001,B904
from __future__ import annotations

from typing import (
    Any,
    List,
    Optional,
    TypedDict,
)

import instructor
from openai import OpenAI

from llm_graphs.agent.step import (
    generate_seed_graph,
    new_graph_from_feedback,
    rate_graph,
)
from llm_graphs.draw_with_pyvis import draw_with_pyvis
from llm_graphs.models import (
    Feedback,
    KnowledgeGraph,
)

GraphDict = TypedDict('GraphDict', {'graph': KnowledgeGraph, 'rating': Optional[Feedback]})

GPT_4O = 'gpt-4o'
GPT_3_5_TURBO = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4'

DEFAULT_MEANING_STR = (
    '- the resulting graph will be visually appealing and give a good global understanding of the structure of the book it explains.\n'
    '- The graph will focus on the concepts and relation between the characters and/or the concepts in the book, not tell the story of the book\n'
    '- Every link description should be of the form close to "<link_description> represent  <reason>" you are allowed not to follow exactly this pattern though\n'
)


def default_goal_str(book_name: str) -> str:
    """Returns the goal string for the graph creator."""
    return f'Generate a graph that will help the reader to understand the structure of the book {book_name}'


class RatingGraphCreator:
    """Class to iteratively generate and rate knowledge graphs for a given book."""

    def __init__(self, book_name: str) -> None:
        """Initialize a new instance of the `RatingGraphCreator` class.

        Parameter
            book_name (str): The name of the book.
        """
        self._client = instructor.from_openai(OpenAI())
        self._graphs_history: List[GraphDict] = []
        self.book_name: str = book_name

    def get_graph(self, ix: int) -> KnowledgeGraph:
        """Return the graph at the given index."""
        if not self._graphs_history:
            raise ValueError('You need to generate a first graph with `generate_initial_graph`')
        if ix > 0:
            raise ValueError('ix should be negative -1 for this graph, -2 for the previous one, etc.')
        try:
            return self._graphs_history[ix]['graph']
        except KeyError:
            raise ValueError(f'No graph at index {ix}')

    def get_rating(self, ix: int) -> Optional[Feedback]:
        """Return the rating at the given index."""
        if not self._graphs_history:
            raise ValueError('You need to generate a first graph with `generate_initial_graph`')
        if ix > 0:
            raise ValueError('ix should be negative -1 for this graph, -2 for the previous one, etc.')
        try:
            return self._graphs_history[ix]['rating']
        except KeyError:
            raise ValueError(f'No rating at index {ix}')

    def generate_initial_graph(self, model: str = GPT_4O) -> KnowledgeGraph:
        """Generate the initial graph."""
        try:
            knowledge_graph = generate_seed_graph(
                model=model,
                goal_str=default_goal_str(self.book_name),
                meaning_str=DEFAULT_MEANING_STR,
            )
            self._graphs_history.append({'graph': knowledge_graph, 'rating': None})
        except Exception as e:
            raise RuntimeError(f'Failed to generate initial graph: {e}')
        return self.get_graph(-1)

    def _rate_graph_from_ix(self, ix: int, model: str = GPT_3_5_TURBO) -> Feedback:
        """Rate the graph."""
        knowledge_graph = self.get_graph(ix)
        return rate_graph(
            model=model,
            goal_str=default_goal_str(self.book_name),
            meaning_str=DEFAULT_MEANING_STR,
            knowledge_graph=knowledge_graph,
        )

    def rate_this_graph(self, model: str = GPT_3_5_TURBO) -> None:
        """Rate the last generated graph."""
        rate_graph = self._rate_graph_from_ix(-1, model)
        self._graphs_history[-1]['rating'] = rate_graph

    def generate_new_graph_from_feedback(self, model: str = GPT_4O) -> KnowledgeGraph:
        """Generate a new graph based on the feedback from the last graph."""

        last_knowledge_graph = self.get_graph(-1)
        last_feedback: Optional[Feedback] = self._graphs_history[-1]['rating']
        if not last_feedback:
            raise ValueError('You need to rate the last graph before generating a new one')
        new_knowledge_graph = new_graph_from_feedback(
            model=model,
            goal_str=default_goal_str(self.book_name),
            meaning_str=DEFAULT_MEANING_STR,
            last_knowledge_graph=last_knowledge_graph,
            last_feedback=last_feedback,
        )
        self._graphs_history.append({'graph': new_knowledge_graph, 'rating': None})
        return new_knowledge_graph

    def rate_and_generate(self, model_rating: str = GPT_3_5_TURBO, model_generation: str = GPT_4O) -> KnowledgeGraph:
        """Rate a graph and generate a better graph based on the rating feedback."""
        self.rate_this_graph(model_rating)
        self.generate_new_graph_from_feedback(model_generation)
        return self.get_graph(-1)

    def plot(self, ix: int = -1) -> Any:
        """Plot the knowledge graph at the given index."""
        last_knowledge_graph: KnowledgeGraph = self.get_graph(ix)
        return draw_with_pyvis(last_knowledge_graph)
