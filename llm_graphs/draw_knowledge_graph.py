from __future__ import annotations

from typing import Any

from pyvis.network import Network  # mypy: ignore-errors

from llm_graphs.models import KnowledgeGraph


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
