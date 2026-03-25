import os
import json  
import logging
from pyvis.network import Network

logger = logging.getLogger(__name__)


def render_graph(triplets: list) -> str:
    net = Network(
        height="650px", 
        width="100%", 
        bgcolor="#0e1117",     
        font_color="#e0e0e0",
        notebook=False
    )
    options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        },
        "edges": {
            "smooth": {"type": "continuous"}, 
            "color": {"inherit": "both"}
        },
        "nodes": {
            "font": {"size": 14, "face": "Helvetica", "color": "white"}
        }
    }
    
    options_json = json.dumps(options)
    net.set_options(f"""var options = {options_json}""")

    seen_nodes = set()

    for triplet in triplets:
        head = triplet.get('head', '')
        relation = triplet.get('type', '')
        tail = triplet.get('tail', '')

        if head and head not in seen_nodes:
            net.add_node(head, label=head, color="#9d4edd", borderWidth=2, size=20, title=f"Entity: {head}")
            seen_nodes.add(head)
        
        if tail and tail not in seen_nodes:
            net.add_node(tail, label=tail, color="#4895ef", borderWidth=2, size=15, title=f"Entity: {tail}")
            seen_nodes.add(tail)

        if head and tail:
            net.add_edge(head, tail, title=relation, label=relation, color="#4a4e69", width=1.5)

    os.makedirs("data/processed", exist_ok=True)
    path = "data/processed/graph.html"
    net.save_graph(path)

    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return html_content