# disable errors due to fastapi decrators not being recognized
# mypy: disable-error-code="misc"
from __future__ import annotations

from typing import (
    List,
)

from fastapi import (
    Body,
    Depends,
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware

from api.db import (
    SessionLocal,
    User,
    get_db,
)
from api.models import FeedbackReturn, GraphAndFeedback, GraphAndFeedbackReturn, GraphReturn, StringReturn
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


@app.get('/')
async def root() -> StringReturn:
    return StringReturn(output='Hello World', success=True)


@app.get('/ping')
async def ping() -> StringReturn:
    return StringReturn(output='pong', success=True)


@app.post('/book_graph/init')
@app.post('/v1/book_graph/init')
def generate_graph_endpoint(book_name: str = Body(), model_name: str = Body('gpt-4o')) -> GraphReturn:
    graph = generate_seed_graph(model=model_name, goal_str=default_goal_str(book_name), meaning_str=DEFAULT_MEANING_STR)
    return GraphReturn(output=graph, success=True)


@app.post('/book_graph/rate')
@app.post('/book_graph/v1/rate')
def rate_graph_endpoint(
    book_name: str = Body(),
    graph: KnowledgeGraph = Body(),
    model_name: str = Body('gpt-4o'),
    num_ratings: int = Body(1),
) -> FeedbackReturn:
    if num_ratings != 1:
        raise HTTPException(status_code=501, detail='Only one rating is supported at the moment')
    rating = rate_graph(
        model=model_name,
        goal_str=default_goal_str(book_name),
        meaning_str=DEFAULT_MEANING_STR,
        knowledge_graph=graph,
    )
    return FeedbackReturn(output=[rating], success=True)


@app.post('/book_graph/improve')
@app.post('/book_graph/v1/improve')
def improve_graph_endpoint(
    book_name: str = Body(),
    graph: KnowledgeGraph = Body(),
    feedbacks: List[Feedback] = Body(),
    model_name: str = Body('gpt-4o'),
) -> GraphReturn:
    new_graph = new_graph_from_feedback(
        model=model_name,
        goal_str=default_goal_str(book_name),
        meaning_str=DEFAULT_MEANING_STR,
        last_knowledge_graph=graph,
        last_feedbacks=feedbacks,
    )
    return GraphReturn(output=new_graph, success=True)


@app.post('/book_graph/rate_and_improve')
@app.post('/book_graph/v1/rate_and_improve')
def rate_and_improve_endpoint(
    book_name: str = Body(),
    graph: KnowledgeGraph = Body(),
    model_name: str = Body('gpt-4o'),
    num_ratings: int = Body(1),
) -> GraphAndFeedbackReturn:
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
    return GraphAndFeedbackReturn(
        output=GraphAndFeedback(graph=new_graph, feedbacks=[feedback]),
        success=True
        )


@app.get('/users')
@app.get('/v1/users')
def create_user(db: SessionLocal = Depends(get_db)) -> StringReturn:
    users: List[User] = db.query(User).all()
    for user in users:
        print(user.username)
    return StringReturn(output='User created', success=True)
