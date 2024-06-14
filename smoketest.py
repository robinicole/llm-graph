from __future__ import annotations

from llm_graphs.agent.rating_agent import RatingGraphCreator

rc = RatingGraphCreator('Test Book')
rc.generate_initial_graph(model='gpt-3.5-turbo')
rc.rate_and_generate(model_for_generation='gpt-3.5-turbo', model_for_rating='gpt-3.5-turbo', num_ratings=2)
