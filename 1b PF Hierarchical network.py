#%%
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#%% --------------------------------------------------------
# Draw Kamada-Kawai layout and report unconnected nodes/edges THIS block can be run without previous blocks
#------------------------------------------------------------

# --- Load edges ---
edges_file = r"\\Thesis code\edges_updated.csv"
edges_df = pd.read_csv(edges_file)

# --- Load nodes ---
nodes_file = r"\\Thesis code\manual_nodes_updated.csv"
nodes_df = pd.read_csv(nodes_file)

# Dictionary: node -> type
node_types = nodes_df.set_index("Node")["Type"].to_dict()

# --- Build undirected graph ---
G = nx.Graph()

# Add nodes with type info
for node, t in node_types.items():
    G.add_node(node, node_type=t)

# Track invalid edges
invalid_edges = []

for _, row in edges_df.iterrows():
    src, tgt = row["Source"], row["Target"]

    if src in G.nodes and tgt in G.nodes:
        G.add_edge(src, tgt, edge_name=row["Edge_Name"], edge_type=row["Edge_Type"])
    else:
        invalid_edges.append((src, tgt, row["Edge_Name"], row["Edge_Type"]))

# --- Find isolated nodes (no edges) ---
isolated_nodes = list(nx.isolates(G))

# --- Layout (Kamada-Kawai) ---
pos = nx.kamada_kawai_layout(G, scale=10)

# Node color mapping
color_map = {
    "big_busbar": "orange",
    "eln": "lightblue",
}

node_colors = []
for n, d in G.nodes(data=True):
    node_colors.append(color_map.get(d.get("node_type", "unknown"), "gray"))

# --- Plot ---
plt.figure(figsize=(26, 18))

nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=80)

nx.draw_networkx_labels(G, pos, font_size=6)

plt.title("Kamada-Kawai Network Layout", fontsize=14)
plt.axis("off")
plt.show()

# --- Print diagnostics ---
print("=== Graph Diagnostics ===")
print(f"Total nodes: {len(G.nodes)}")
print(f"Total edges: {len(G.edges)}")
print()

if isolated_nodes:
    print(f"Isolated nodes (no connections): {isolated_nodes}\n")
else:
    print("No isolated nodes found.\n")

if invalid_edges:
    print("Edges with missing nodes:")
    for e in invalid_edges:
        print(f"  {e}")
else:
    print("No invalid edges found.")
































## these values work: instead of making tuples, directly put the value in the node congestion vertical file





# %%
