from __future__ import annotations

import unittest

from llm_graphs.agent.prompts import (
    system_graph_creator,
    user_generate_graph,
    user_improve_from_feedback,
    user_rate_graph,
)
from llm_graphs.models import (
    Feedback,
    KnowledgeGraph,
)


class TestPrompts(unittest.TestCase):
    def test_system_graph_creator(self) -> None:
        expected_message = {
            'role': 'system',
            'content': 'You are an expert in summarizing data into visually appealing knowledge graphs.',
        }
        result = system_graph_creator()
        assert result == expected_message

    def test_user_generate_graph(self) -> None:
        goal_str = 'Generate a graph that will help the reader understand the structure of the book.'
        meaning_str = '- The resulting graph will be visually appealing and give a good global understanding of the structure of the book it explains.'
        user_generate_graph(goal_str, meaning_str)

    def test_user_rate_graph(self) -> None:
        goal_str = 'Generate a graph that will help the reader understand the structure of the book.'
        meaning_str = '- The resulting graph will be visually appealing and give a good global understanding of the structure of the book it explains.'
        knowledge_graph = KnowledgeGraph(
            nodes=[],
            links=[],
            name='Test Graph',
            reasoning='This is a test graph.',
        )
        user_rate_graph(goal_str, meaning_str, knowledge_graph)

    def test_user_improve_from_feedback(self) -> None:
        goal_str = 'Generate a graph that will help the reader understand the structure of the book.'
        meaning_str = '- The resulting graph will be visually appealing and give a good global understanding of the structure of the book it explains.'
        last_knowledge_graph = KnowledgeGraph(
            nodes=[],
            links=[],
            name='Test Graph',
            reasoning='This is a test graph.',
        )
        last_feedback = Feedback(rating=5, opinion='Needs improvement in structure.')
        user_improve_from_feedback(goal_str, meaning_str, last_knowledge_graph, last_feedback)


if __name__ == '__main__':
    unittest.main()
