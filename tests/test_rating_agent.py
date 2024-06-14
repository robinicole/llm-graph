from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from llm_graphs.agent.rating_agent import RatingGraphCreator
from llm_graphs.models import (
    Feedback,
    KnowledgeGraph,
)


class TestRatingGraphCreator(unittest.TestCase):
    def setUp(self) -> None:
        self.book_name = 'Test Book'
        self.mock_client = MagicMock()
        self.creator = RatingGraphCreator(self.book_name)
        self.creator._client = self.mock_client
        self.mock_graph = KnowledgeGraph(nodes=[], links=[], name='Test Graph', reasoning='Test reasoning')
        self.mock_feedback = Feedback(rating=8, opinion='Test opinion')

    def test_generate_initial_graph(self) -> None:
        self.creator._client.chat.completions.create.return_value = self.mock_graph  # type: ignore
        graph = self.creator.generate_initial_graph()
        self.creator._client.chat.completions.create.assert_called_once()  # type: ignore
        assert isinstance(graph, KnowledgeGraph)
        args, kwargs = self.creator._client.chat.completions.create.call_args  # type: ignore
        assert len(kwargs['messages']) == 2

    def test_get_graph(self) -> None:
        self.creator._graphs_history = [{'graph': self.mock_graph, 'rating': None}]
        assert self.creator.get_graph(-1) == self.mock_graph

        with self.assertRaises(ValueError):
            self.creator.get_graph(1)

        self.creator._graphs_history = []
        with self.assertRaises(ValueError):
            self.creator.get_graph(-1)

    def test_get_rating(self) -> None:
        self.creator._graphs_history = [{'graph': self.mock_graph, 'rating': [self.mock_feedback]}]
        assert self.creator.get_rating(-1) == [self.mock_feedback]

        with self.assertRaises(ValueError):
            self.creator.get_rating(1)

        self.creator._graphs_history = []
        with self.assertRaises(ValueError):
            self.creator.get_rating(-1)

    # @patch('your_module.rate_graph', return_value=Feedback(text="Test feedback"))
    # def test_rate_this_graph(self, mock_rate_graph):

    # @patch('your_module.new_graph_from_feedback', return_value=KnowledgeGraph(nodes=[], edges=[]))
    # def test_generate_new_graph_from_feedback(self, mock_new_graph_from_feedback):

    #     with self.assertRaises(ValueError):

    def test_rate_and_generate_fail_without_initial_graph(self) -> None:
        with self.assertRaises(ValueError):
            self.creator.rate_and_generate()

    def test_rate_and_generate(self) -> None:
        # add an initial graph
        self.creator._graphs_history = [{'graph': self.mock_graph, 'rating': None}]
        self.mock_client.chat.completions.create = MagicMock(side_effect=(self.mock_feedback, self.mock_graph))
        self.creator.rate_and_generate()
        instructor_args = self.mock_client.chat.completions.create.call_args_list
        assert len(instructor_args) == 2, instructor_args
        rating_user_prompt = instructor_args[0].kwargs['messages'][1]['content']
        assert (
            'Please rate the graph above from 0 to 10 and give feedback on how it can be improved' in rating_user_prompt
        )
        assert self.mock_graph.model_dump_json() in rating_user_prompt
        generate_from_rating_prompt = instructor_args[1].kwargs['messages'][1]['content']
        assert '- Rating: 8/10 Opinion: Test opinion' in generate_from_rating_prompt


if __name__ == '__main__':
    unittest.main()
