
#%%

#%%
import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# --- Paths ---
nodes_csv = r"C:\\Thesis code\manual_nodes_updated.csv"
edges_csv = r"C:\\Thesis code\edges_updated.csv"
loads_combined_csv = r"C:\\Thesis code\loads_combined.csv"
finished_loads_folder = r"C:\\Finished loads"

output_node_values = r"C:\\Thesis code\node_values_output.csv"
output_graph_png = r"C:\\Thesis code\network_graph.png"
SAVE_FIG = False

# --- Helper: Resolve file paths using glob ---
def resolve_file_path(basename: str, folder: str) -> list:
    """Return list of matching file paths for basename (supports wildcards)."""
    pattern = os.path.join(folder, f"{basename}.csv")
    return glob.glob(pattern)

# --- Helper: Get value at row 5587 ---
def get_reqflex_minus_actflex_at_row(path: str, row_index: int = 5611) -> float:
    """
    Read CSV and return the difference (reqflex - actflex) at a specific row index.
    If the index is out of bounds or values are NaN, return 0.0.
    """
    df = pd.read_csv(path)
    if row_index >= len(df):
        print(f"[WARNING] Row {row_index} out of bounds for file: {path}")
        return 0.0

    reqflex_col = df.columns[-2]
    actflex_col = df.columns[-1]

    reqflex = pd.to_numeric(df.loc[row_index, reqflex_col], errors="coerce")
    actflex = pd.to_numeric(df.loc[row_index, actflex_col], errors="coerce")

    if pd.isna(reqflex) or pd.isna(actflex):
        return 0.0

    return actflex - reqflex # clip negative differences to zero

# --- Step 1: Read nodes and edges ---
nodes = pd.read_csv(nodes_csv).iloc[:, 0].astype(str).str.strip().tolist()
edges_df = pd.read_csv(edges_csv)
edges = [(str(row[0]).strip(), str(row[1]).strip()) for _, row in edges_df.iloc[:, :2].iterrows()]

# --- Step 2: Node -> filename mapping ---
loads_map_df = pd.read_csv(loads_combined_csv)
node_to_filename = dict(zip(loads_map_df.iloc[:, 0].str.strip(), loads_map_df.iloc[:, 1].str.strip()))

# --- Step 3: Compute node values ---
node_values = {}
missing_mappings, missing_files = [], []

for node in nodes:
    basename = node_to_filename.get(node)
    if not basename:
        node_values[node] = 0.0
        missing_mappings.append(node)
        print(f"[INFO] No mapping for node: {node}")
        continue

    file_paths = resolve_file_path(basename, finished_loads_folder)
    if not file_paths:
        node_values[node] = 0.0
        missing_files.append(basename)
        print(f"[WARNING] No file found for mapping: {basename}")
        continue

    print(f"[INFO] Node '{node}' -> Files found: {', '.join(os.path.basename(fp) for fp in file_paths)}")

    total_diff = 0.0
    for fp in file_paths:
        file_diff = get_reqflex_minus_actflex_at_row(fp)
        print(f"    [DETAIL] Processing file: {os.path.basename(fp)}, Shortage (reqflex - actflex) = {file_diff}")
        total_diff += file_diff

    node_values[node] = total_diff
    print(f"[RESULT] Node '{node}' total shortage = {total_diff}")

# Save missing info for debugging
pd.DataFrame(missing_mappings, columns=["Missing Nodes"]).to_csv("missing_nodes.csv", index=False)
pd.DataFrame(missing_files, columns=["Missing Files"]).to_csv("missing_files.csv", index=False)

#%%
# --- Step 4: Build graph ---
G = nx.Graph()
for node in nodes:
    G.add_node(node, value=node_values.get(node, 0.0))
G.add_edges_from(edges)

# --- Step 5: Draw graph ---
values = [G.nodes[n]["value"] for n in G.nodes()]

manual_max = 0.20  # <-- your chosen upper bound
vmin = -0.20         # or another lower bound you prefer
vmax = manual_max  # instead of max(values)

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(16, 10))
pos = nx.kamada_kawai_layout(G)
for k in pos:
    pos[k] = pos[k] * 2.5

nx.draw_networkx_nodes(G, pos, node_color=values, cmap=plt.cm.RdYlGn, vmin=vmin, vmax=vmax, node_size=80, ax=ax)

# Draw circles around nodes
nx.draw_networkx_nodes(G, pos, node_color='none', edgecolors='black', node_size=81, linewidths=0.1, ax=ax)

nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)

# Optional: Label nodes with high values
labels = {n: f"{v:.1f}" for n, v in node_values.items() if v > 10}
nx.draw_networkx_labels(G, pos, labels, font_size=6)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Percentual flexibility in [0,1] interval", fontsize=10)
cbar.ax.tick_params(labelsize=8)

ax.set_title("Network Graph with Node Values", fontsize=14, pad=20)
ax.axis("off")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(output_graph_png, dpi=300)
plt.show()

# --- Step 6: Save node values ---


# %%
