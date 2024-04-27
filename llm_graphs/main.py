"""The main function."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

# Pydantic to specify LLM output schema
import instructor
from openai import OpenAI

# Plotting utils
from pyvis.network import Network

from llm_graphs.models import KnowledgeGraph


@lru_cache(maxsize=None)
def get_knowledge_graph_object(
    book_title: str,
    model: str = 'gpt-4',
) -> KnowledgeGraph:
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
        The OpenAI API model to use, by default 'gpt-4'
    client : OpenAI, optional
        The OpenAI client to use, by default `instructor.from_openai(OpenAI())`

    Returns
    -------
    KnowledgeGraph
        The generated graph representing the book.
    """
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model=model,
        response_model=KnowledgeGraph,
        messages=[
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
        ],
    )


def draw_kg_with_pyvis(knowledge_graph: KnowledgeGraph) -> Any:  # noqa: D417,ANN401
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
