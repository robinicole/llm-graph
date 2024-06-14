# disable errors due to fastapi decrators not being recognized
# mypy: disable-error-code="misc"
from __future__ import annotations

from typing import (
    Any,
    List,
)

from fastapi import (
    Body,
    Depends,
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.db import (
    SessionLocal,
    User,
    get_db,
)
from llm_graphs.agents.rating_agent import (
    DEFAULT_MEANING_STR,
    default_goal_str,
)
from llm_graphs.models import (
    Feedback,
    KnowledgeGraph,
)
from llm_graphs.step import (
    generate_seed_graph,
    new_graph_from_feedback,
    rate_graph,
)

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class GenericReturn(BaseModel):
    output: Any
    success: bool


@app.get('/')
async def root() -> GenericReturn:
    return GenericReturn(output='Hello World', success=True)


@app.get('/ping')
async def ping() -> GenericReturn:
    return GenericReturn(output='pong', success=True)


@app.post('/book_graph/init')
@app.post('/v1/book_graph/init')
def generate_graph_endpoint(book_name: str = Body(), model_name: str = Body('gpt-4o')) -> GenericReturn:
    graph = generate_seed_graph(model=model_name, goal_str=default_goal_str(book_name), meaning_str=DEFAULT_MEANING_STR)
    return GenericReturn(output=graph, success=True)


@app.post('/book_graph/rate')
@app.post('/book_graph/v1/rate')
def rate_graph_endpoint(
    book_name: str = Body(),
    graph: KnowledgeGraph = Body(),
    model_name: str = Body('gpt-4o'),
    num_ratings: int = Body(1),
) -> GenericReturn:
    if num_ratings != 1:
        raise HTTPException(status_code=501, detail='Only one rating is supported at the moment')
    rating = rate_graph(
        model=model_name,
        goal_str=default_goal_str(book_name),
        meaning_str=DEFAULT_MEANING_STR,
        knowledge_graph=graph,
    )
    return GenericReturn(output=[rating], success=True)


@app.post('/book_graph/improve')
@app.post('/book_graph/v1/improve')
def improve_graph_endpoint(
    book_name: str = Body(),
    graph: KnowledgeGraph = Body(),
    feedbacks: List[Feedback] = Body(),
    model_name: str = Body('gpt-4o'),
) -> GenericReturn:
    new_graph = new_graph_from_feedback(
        model=model_name,
        goal_str=default_goal_str(book_name),
        meaning_str=DEFAULT_MEANING_STR,
        last_knowledge_graph=graph,
        last_feedbacks=feedbacks,
    )
    return GenericReturn(output=new_graph, success=True)


@app.post('/book_graph/rate_and_improve')
@app.post('/book_graph/v1/rate_and_improve')
def rate_and_improve_endpoint(
    book_name: str = Body(),
    graph: KnowledgeGraph = Body(),
    model_name: str = Body('gpt-4o'),
    num_ratings: int = Body(1),
) -> GenericReturn:
    if num_ratings != 1:
        raise HTTPException(status_code=501, detail='Only one rating is supported at the moment')
    feedback = rate_graph(
        model=model_name,
        goal_str=default_goal_str(book_name),
        meaning_str=DEFAULT_MEANING_STR,
        knowledge_graph=graph,
    )
    new_graph = new_graph_from_feedback(
        model=model_name,
        goal_str=default_goal_str(book_name),
        meaning_str=DEFAULT_MEANING_STR,
        last_knowledge_graph=graph,
        last_feedbacks=[feedback],
    )
    return GenericReturn(output={'new_graph': new_graph, 'feedbacks': [feedback]}, success=True)


@app.get('/users')
@app.get('/v1/users')
def create_user(db: SessionLocal = Depends(get_db)) -> GenericReturn:
    users: List[User] = db.query(User).all()
    for user in users:
        print(user.username)
    return GenericReturn(output='User created', success=True)
