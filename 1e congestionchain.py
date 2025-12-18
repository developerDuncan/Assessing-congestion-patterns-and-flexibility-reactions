
#%% --------------------------------------------------------
# STEP 1: Build base congestion mapping (only direct parent edge)
#         Save CSV so you can manually add/edit congestion components
#------------------------------------------------------------

import pandas as pd
from collections import deque

# Ensure node labels are strings (important for ELN numeric nodes)
edges_df["Source"] = edges_df["Source"].astype(str)
edges_df["Target"] = edges_df["Target"].astype(str)
nodes_df["Node"] = nodes_df["Node"].astype(str)

# Define root nodes (no congestion component)
root_nodes = ["xxxx", "xxxx"]

# Store results: node -> direct congestion component
direct_congestion = {root: None for root in root_nodes}

# BFS traversal to assign direct congestion edge
queue = deque(root_nodes)
visited = set(root_nodes)

while queue:
    current = queue.popleft()

    for neighbor in G.neighbors(current):
        if neighbor not in visited:
            edge_data = G.get_edge_data(current, neighbor)
            edge_name = edge_data.get("edge_name") if edge_data else None

            # Direct congestion = edge name only (not inherited chain)
            direct_congestion[neighbor] = edge_name

            visited.add(neighbor)
            queue.append(neighbor)

# Build DataFrame (one row per node, Direct_Congestion can later hold multiple entries)
direct_congestion_df = pd.DataFrame([
    {"Node": node, "Direct_Congestion": direct_congestion.get(node)}
    for node in G.nodes
])

# Save to CSV so you can manually edit/add multiple congestion components per node
base_file = r"\\Thesis code\base_congestion_components.csv"
direct_congestion_df.to_csv(base_file, index=False)

print("Base congestion file saved to:", base_file)
print(direct_congestion_df.head(20))


#-----------------------------------------------------------------------


#%% --------------------------------------------------------
# STEP 2: Rebuild full hierarchy using manually adjusted congestion file
#         (Supports multiple congestion components per node, separated by ;)
#------------------------------------------------------------

from collections import deque

# Load the manually adjusted file
adjusted_file = r"\\Thesis code\base_congestion_components.csv"
adjusted_df = pd.read_csv(adjusted_file).astype(str)

# Parse congestion components: split on ";"
manual_map = {
    row["Node"]: [c.strip() for c in str(row["Direct_Congestion"]).split(";") if c.strip() and c.strip().lower() != "nan"]
    for _, row in adjusted_df.iterrows()
}

# Propagate through graph with inheritance
hierarchy_map = {root: [] for root in root_nodes}

queue = deque(root_nodes)
visited = set(root_nodes)

while queue:
    current = queue.popleft()
    parent_congestion = hierarchy_map[current]

    for neighbor in G.neighbors(current):
        if neighbor not in visited:
            # Own congestion from manual file (possibly multiple entries)
            own_congestion = manual_map.get(neighbor, [])

            # Full congestion = inherited + own
            hierarchy_map[neighbor] = parent_congestion + own_congestion

            visited.add(neighbor)
            queue.append(neighbor)

# Build final DataFrame
hierarchy_df = pd.DataFrame([
    {
        "Node": node,
        "Full_Congestion_Path": " > ".join(comp) if comp else None
    }
    for node, comp in hierarchy_map.items()
])

# Save final propagated congestion
final_file = r"\\Thesis code\final_congestion_components.csv"
hierarchy_df.to_csv(final_file, index=False)

print("Final hierarchical congestion file saved to:", final_file)
print(hierarchy_df.head(20))


