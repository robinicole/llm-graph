from __future__ import annotations

from typing import Any

from fastapi import (
    Body,
    FastAPI,
)
from pydantic import BaseModel

from llm_graphs.agent.rating_agent import (
    DEFAULT_MEANING_STR,
    default_goal_str,
)
from llm_graphs.agent.step import generate_seed_graph

app = FastAPI()


class GenericReturn(BaseModel):
    output: Any
    success: bool


@app.get('/')
async def root() -> GenericReturn:
    return GenericReturn(output='Hello World', success=True)


@app.post('/book_graph/init')
def generate_graph(book_name: str = Body(), model_name: str = Body('gpt-4o')) -> GenericReturn:
    graph = generate_seed_graph(model=model_name, goal_str=default_goal_str(book_name), meaning_str=DEFAULT_MEANING_STR)
    return GenericReturn(output=graph, success=True)
