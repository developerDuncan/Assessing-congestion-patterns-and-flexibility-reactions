#%%
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

# File paths
loads_csv = r"C:\\Thesis code\loads_combined.csv"
edges_csv = r"C:\\Thesis code\edges_updated.csv"
finished_loads_dir = r"C:\\Finished loads"


# %%
# Read loads_combined.csv
loads_df = pd.read_csv(loads_csv, header=None, skiprows=1, usecols=[0,1], dtype=str)
loads_df.columns = ['node', 'file_name']

# Initialize a dictionary for node loads
node_loads = {str(row['node']): None for _, row in loads_df.iterrows()}


# %%
all_files = [f for f in os.listdir(finished_loads_dir) if f.endswith('.csv')]

for _, row in loads_df.iterrows():
    node = str(row['node'])
    file_name = str(row['file_name']).strip()
    
    # Extract the important part
    if ',' in file_name:
        file_name = file_name.split(',')[-1]
    if file_name.startswith('*'):
        file_name = file_name[1:]
    
    # Search for a file that contains this part
    matching_files = [f for f in all_files if file_name in f]
    
    if matching_files:
        matched_file = matching_files[0]
        print(f"Node {node}: file part '{file_name}' → FOUND as '{matched_file}'")
    else:
        print(f"Node {node}: file part '{file_name}' → NOT FOUND")


# %%
for idx, row in loads_df.iterrows():
    node = str(row['node'])
    file_name = str(row['file_name']).strip()
    
    # Apply the same extraction rules as before
    if ',' in file_name:
        file_name = file_name.split(',')[-1]
    if file_name.startswith('*'):
        file_name = file_name[1:]
    
    # Find matching file
    matching_files = [f for f in all_files if file_name in f]
    
    if matching_files:
        matched_file = matching_files[0]
        file_path = os.path.join(finished_loads_dir, matched_file)
        try:
            df = pd.read_csv(file_path, header=None, sep=',')  # No header assumed
            
            # Determine which column to sum
            if df.shape[1] == 7:
                col_index = 5  # 6th column
            elif df.shape[1] == 6:
                col_index = 4  # 5th column
            else:
                print(f"Node {node}: file '{matched_file}' has unexpected number of columns ({df.shape[1]})")
                loads_df.at[idx, 'total_load'] = float('nan')
                continue
            
            # Sum the selected column (handle decimal commas and non-numeric values)
            col_values = df.iloc[:, col_index].astype(str).str.strip()
            col_values = col_values.str.replace(',', '.')
            total = pd.to_numeric(col_values, errors='coerce').fillna(0).sum()
            
            loads_df.at[idx, 'total_load'] = total
            print(f"Node {node}: file '{matched_file}' → total load = {total}")
        except Exception as e:
            print(f"Node {node}: failed to read '{matched_file}' → {e}")
    else:
        print(f"Node {node}: no matching file to sum loads")
# %%
output_csv = r"C:\\Thesis code\node_current_values.csv"

loads_df.to_csv(output_csv, index=False)
print(f"Node loads saved to: {output_csv}")

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# --- Load edges ---
edges_file = r"C:\\Thesis code\edges_updated.csv"
edges_df = pd.read_csv(edges_file)

# --- Load manual nodes ---
manual_nodes_file = r"C:\\Thesis code\manual_nodes_updated.csv"
manual_nodes_df = pd.read_csv(manual_nodes_file)

# --- Load node values ---
node_values_file = r"C:\\Thesis code\node_current_values.csv"
values_df = pd.read_csv(node_values_file, header=None, usecols=[0, 2], names=['Node', 'Value'])

# Create dictionary: Node -> Value
value_dict = dict(zip(values_df['Node'], values_df['Value']))

#%%
print(value_dict)
#%%)

# --- Build undirected graph ---
G = nx.Graph()

# Add nodes with type and value
for _, row in manual_nodes_df.iterrows():
    node = row['Node']
    node_type = row.get('Type', 'unknown')
    value = value_dict.get(node, 0)  # get value if exists, else 0
    try:
        value = float(value)
    except:
        value = 0
    G.add_node(node, node_type=node_type, value=value)

# Track invalid edges
invalid_edges = []

for _, row in edges_df.iterrows():
    src, tgt = row.iloc[0], row.iloc[1]  # assuming source in col 0, target in col 1
    edge_name = row.iloc[2] if len(row) > 2 else ""
    edge_type = row.iloc[3] if len(row) > 3 else ""

    if src in G.nodes and tgt in G.nodes:
        G.add_edge(src, tgt, edge_name=edge_name, edge_type=edge_type)
    else:
        invalid_edges.append((src, tgt, edge_name, edge_type))

# Find isolated nodes
isolated_nodes = list(nx.isolates(G))

# Layout
pos = nx.kamada_kawai_layout(G, scale=10)

# Node colors based on log-scaled value (Viridis)
# Node colors based on linear value (Viridis)
values = np.array([G.nodes[n]['value'] for n in G.nodes()])
values = np.nan_to_num(values, nan=0.0)

# Normalize linearly between min and max
norm = plt.Normalize(vmin=values.min(), vmax=values.max())
colors = cm.viridis(norm(values))

# Plot
# Plot
plt.figure(figsize=(26, 18))
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=100)
nx.draw_networkx_labels(G, pos, font_size=6)

# Colorbar fix
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array(values)
plt.colorbar(sm, ax=plt.gca(), label="Node Value")  # explicitly specify axes

plt.title("Kamada-Kawai Network Layout with Node Values", fontsize=14)
plt.axis("off")
plt.show()

# %%
