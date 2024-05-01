"""The main function."""
from __future__ import annotations

from functools import lru_cache
from typing import (
    Any,
    List,
)

# Pydantic to specify LLM output schema
import instructor
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam  # noqa: TCH002

# Plotting utils
from pyvis.network import Network

from llm_graphs.models import KnowledgeGraph


def _from_prompt(
    messages: List[ChatCompletionMessageParam],
    model: str = 'gpt-4',
) -> KnowledgeGraph:
    """Return a KnowledgeGraph object for a given book title using the OpenAI API.

    The function first generates some text using the OpenAI API, then uses the
    Pydantic model `KnowledgeGraph` to parse the resulting text as a graph.
    The function is memoized using functools.lru_cache to avoid generating
    the same graph multiple times.

    Parameters
    ----------
    messages : List[ChatCompletionMessageParam]
        The messages to send to the OpenAI API
    model : str, optional
        The OpenAI API model to use, by default 'gpt-4'

    Returns
    -------
    KnowledgeGraph
        The generated graph representing the book.
    """
    if model not in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']:
        raise ValueError('Model must be either gpt-3.5-turbo or gpt-4')
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model=model,
        response_model=KnowledgeGraph,
        messages=messages,
    )


@lru_cache(maxsize=None)
def from_book_summary(book_title: str, model: str = 'gpt-4') -> KnowledgeGraph:
    """Return a KnowledgeGraph object for a given book title using the OpenAI API.

    The function first generates some text using the OpenAI API, then uses the
    Pydantic model `KnowledgeGraph` to parse the resulting text as a graph.
    The function is memoized using functools.lru_cache to avoid generating
    the same graph multiple times.

    Parameters
    ----------
    book_title : str
        The title of the book to summarize.
    model : str, optional
        The OpenAI API model to use, by default 'gpt-4'.

    Returns
    -------
    KnowledgeGraph
        The generated graph representing the book.
    """
    messages: List[ChatCompletionMessageParam] = [
        {
            'role': 'system',
            'content': 'You are an avid reader and you summarize the books in knowledge graphs and make it entertaining',  # noqa: E501
        },
        {
            'role': 'user',
            'content': f'''Summarize the book {book_title} in the graph given as type
Each of the nodes into the graph represent one main concept of the graph
Each of the link represent a link between two main concept
the graph should  have 10 nodes and 20 links.
the resulting graph should be visually appealing and give a good global understanding of  the boo it summarizes.
Take some time and reason step by step before creating the graph object to make a graph that will be easy to display
A node should not have self loop and there should not be a loop between two nodes
The graph should give to the reader a good summary of the book''',
        },
    ]
    return _from_prompt(messages=messages, model=model)


def draw_with_pyvis(knowledge_graph: KnowledgeGraph) -> Any:  # noqa: D417,ANN401
    """Draws a knowledge graph using Pyvis library.

    Parameters
    ----------
        knowledge_graph (KnowledgeGraph): The knowledge graph object to visualize.

    Returns
    -------
        str: The path to the HTML file where the network is displayed.
    """
    net = Network(notebook=True, directed=True, width='800px', height='600px')

    # Add nodes to the network
    for node in knowledge_graph.nodes:
        net.add_node(node.node_id, label=node.name, title=node.description.replace('.', '.\n'), shape='box')

    # Add edges to the network
    for link in knowledge_graph.links:
        net.add_edge(link.node_id_from, link.node_id_to, label=link.name, title=link.description.replace('.', '.\n'))

    # Set layout options
    net.barnes_hut(gravity=-1000, overlap=100)

    # Display the network
    return net.show(f'{knowledge_graph.name}.html')


if __name__ == '__main__':
    # receive command line args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('book_title', type=str, help='The title of the book to summarize.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='The OpenAI API model to use.')
    args = parser.parse_args()
    # generate the knowledge graph
    knowledge_graph = from_book_summary(book_title=args.book_title, model=args.model)
    draw_with_pyvis(knowledge_graph)
