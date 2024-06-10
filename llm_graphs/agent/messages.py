from __future__ import annotations

from openai.types.chat import ChatCompletionMessageParam  # noqa: TCH002

from llm_graphs.models import (
    Feedback,
    KnowledgeGraph,
)


def system_graph_creator() -> ChatCompletionMessageParam:
    """Return the system message for the graph creator."""
    return {
        'role': 'system',
        'content': 'You are an expert in summarizing data into visually appealing knowledge graphs.',
    }


def user_generate_graph(goal_str: str, meaning_str: str) -> ChatCompletionMessageParam:
    """Return the user message for the graph creator."""
    return {
        'role': 'user',
        'content': f'''
# Goal
{goal_str}
# Graph meaning
{meaning_str}
# Graph structure
for the edges:
- idFrom and idTo are the ids of the nodes from where the link starts and where the link is directed
- Each node represents a concept in the book
- If the graph is too complex to be represented in 2D you are allowed to remove some edges and nodes
- If you have a link between two node ids that do not exist you have failed your task
- If there is an isolated node you have failed your task
- The graph should not have isolated components
- A node should not have a self-loop and there should not be a loop between two nodes
- The graph should ideally have 10 nodes and 20 edges but be flexible
Take some time and reason step by step before creating the graph object to make a graph that will be easy to display
- You should describe your thought process in the reasoning field of the graph
- Take a deep breath and work through this step by step and make sure you have the right answer
''',
    }


def user_rate_graph(
    goal_str: str,
    meaning_str: str,
    knowledge_graph: KnowledgeGraph,
) -> ChatCompletionMessageParam:
    """Return the message to rate the graph."""
    return {
        'role': 'user',
        'content': f'''
Graph json
{knowledge_graph.model_dump_json()}

The json above represents a graph that was generated and is supposed to follow the criteria below
<criteria>
# Goal
{goal_str}
# Graph meaning
{meaning_str}
</criteria>
Please rate the graph above from 0 to 10 and give feedback on how it can be improved
You should give extra marks to graphs that explain the concepts of the book and help the reader interpret the book as they are reading it
You should penalize graphs with loops between two nodes and give extra marks to graphs with non-standard structures
''',
    }


def user_improve_from_feedback(
    goal_str: str,
    meaning_str: str,
    last_knowledge_graph: KnowledgeGraph,
    last_feedback: Feedback,
) -> ChatCompletionMessageParam:
    """Return the message to improve the graph based in its rating."""
    return {
        'role': 'user',
        'content': f'''
You made the graph below
```
{last_knowledge_graph.model_dump_json()}
```
Following those instructions
# Goal
{goal_str}
# Graph meaning
{meaning_str}
And received the following rating {last_feedback.rating}/10 and this feedback {last_feedback.opinion}. Improve the graph
''',
    }
