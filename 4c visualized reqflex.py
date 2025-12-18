#%%
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- Paths ---
nodes_csv = r"C:\\Thesis code\manual_nodes_updated.csv"
edges_csv = r"C:\\Thesis code\edges_updated.csv"
loads_combined_csv = r"C:\\Thesis code\loads_combined.csv"
finished_loads_folder = r"C:\\Finished loads"

output_node_values = r"C:\Users\\Thesis code\node_values_output.csv"
output_graph_png = r"C:\Users\\Thesis code\network_graph.png"
SAVE_FIG = False

def resolve_file_path(basename: str, folder: str) -> list:
    """Return list of matching file paths for basename (handles '*' wildcard at start or end)."""
    if basename.startswith("*"):  # match files ending with pattern
        suffix = basename[1:]
        matches = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.endswith(suffix + ".csv")]
        return matches
    elif basename.endswith("*"):  # match files starting with pattern
        prefix = basename[:-2]
        matches = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.startswith(prefix) and f.endswith(".csv")]
        return matches
    else:
        file_path = os.path.join(folder, f"{basename}.csv")
        return [file_path] if os.path.exists(file_path) else []



# --- Helper: Sum last column ---
def sum_last_column_from_csv(path: str) -> float:
    """Read CSV and return sum of its last column (numeric)."""
    df = pd.read_csv(path)
    if df.shape[1] < 1:
        return 0.0
    col = df.columns[-2]  # last column
    series = pd.to_numeric(df[col], errors="coerce")
    return float(series.dropna().sum())

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

    # Print all files found for this node
    print(f"[INFO] Node '{node}' -> Files found: {', '.join(os.path.basename(fp) for fp in file_paths)}")

    # Sum across all matching files
    total_sum = 0.0
    for fp in file_paths:
        file_sum = sum_last_column_from_csv(fp)
        print(f"    [DETAIL] Processing file: {os.path.basename(fp)}, Last column sum = {file_sum}")
        total_sum += file_sum

    node_values[node] = total_sum
    print(f"[RESULT] Node '{node}' total value = {total_sum}")




# --- Step 4: Build graph ---
G = nx.Graph()
for node in nodes:
    G.add_node(node, value=node_values.get(node, 0.0))
G.add_edges_from(edges)

# --- Step 5: Draw graph ---
values = [G.nodes[n]["value"] for n in G.nodes()]
vmin, vmax = min(values), max(values)
if vmin == vmax:
    vmin, vmax = 0.0, vmax if vmax > 0 else 1.0

fig, ax = plt.subplots(figsize=(16, 10))
pos = nx.kamada_kawai_layout(G)
for k in pos:
    pos[k] = pos[k] * 2.5

nx.draw_networkx_nodes(G, pos, node_color=values, cmap=plt.cm.Reds, vmin=vmin, vmax=vmax, node_size=80, ax=ax)

# Draw circles around nodes (slightly bigger, no fill, colored edge)
nx.draw_networkx_nodes(
    G, pos,
    node_color='none',       # no fill
    edgecolors='black',      # circle color
    node_size=81,           # bigger than original
    linewidths=0.1,            # thickness of circle
    ax=ax
)

nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
#nx.draw_networkx_labels(G, pos, font_size=6, font_color="black", ax=ax)

sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Summed congestion and required flexibility", fontsize=10)
cbar.ax.tick_params(labelsize=8)

ax.set_title("Summed congestion and required flexibility", fontsize=14, pad=20)
ax.axis("off")
plt.tight_layout()

if SAVE_FIG:
    plt.savefig(output_graph_png, dpi=300)
plt.show()

# --- Step 6: Save node values ---
pd.DataFrame(list(node_values.items()), columns=["Node", "Value"]).to_csv(output_node_values, index=False)
print(f"Node values saved to {output_node_values}")

# %%
