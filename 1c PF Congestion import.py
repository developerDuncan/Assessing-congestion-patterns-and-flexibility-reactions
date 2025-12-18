# %% in this file, the node congestion file is created and later on filled
import pandas as pd
#%%

# Paths
components_file = r"\\Thesis code\final_congestion_components.csv"
output_file = r"\\Thesis code\node_congestion_vertical.csv"

# Load the CSV
components_df = pd.read_csv(components_file)

# Prepare list to build new DataFrame
rows = []

for _, row in components_df.iterrows():
    node = row['Node']
    # First row: node only
    rows.append([node, ''])
    # Split congestion components
    comps = [comp.strip() for comp in str(row['Full_Congestion_Path']).split('>') if comp.strip()]
    # Add components in rows below node
    for comp in comps:
        rows.append(['', comp])
    # Add empty row to separate from next node
    rows.append(['', ''])

# Create DataFrame
vertical_df = pd.DataFrame(rows, columns=['Node', 'Congestion_Component'])

# Save to CSV
vertical_df.to_csv(output_file, index=False)

print("Vertical node congestion components saved to:", output_file)
print(vertical_df.head(20))

# %%

# Paths
vertical_file = output_file  # same CSV we just created
loading_file = r"\\Bureaublad\big.csv" #change this one to the bigger file

# Load vertical CSV
vertical_df = pd.read_csv(vertical_file)

# Strip column names just in case
vertical_df.columns = vertical_df.columns.str.strip()

# Load small2.csv to get time steps (first column starting at row 3)
loading_df = pd.read_csv(loading_file, sep=';', header=None, skiprows=2)

# Extract time steps
time_steps = loading_df.iloc[:, 0].tolist()

# Build new rows
rows = []

# First row: header with 'Time' + all time steps
rows.append(['Time', ''] + list(map(str, time_steps)))

# Append vertical layout (Node / Congestion_Component)
for _, row in vertical_df.iterrows():
    rows.append([row['Node'], row['Congestion_Component']])

# Convert to DataFrame
expanded_df = pd.DataFrame(rows)

# Save back to CSV (overwrite or new file)
expanded_df.to_csv(vertical_file, index=False, header=False, sep=';')

print("Vertical CSV expanded with time steps in the first row saved to:", vertical_file)
print(expanded_df.head(20))

#Till here was only the creation of the specific dataframe so that one can see what elements belong to what specific busbar


#%%
from collections import defaultdict

# File paths
vertical_file = r"\\Bureaublad\Thesis code\node_congestion_vertical.csv"
loading_file = r"\\Bureaublad\big.csv"
output_file = r"\\Thesis code\node_congestion_vertical_filled.csv"

# Load CSVs
loading_df = pd.read_csv(loading_file, header=None, sep=';')
vertical_df = pd.read_csv(vertical_file, header=None, sep=';', dtype=str)  # keep identifiers as strings

# --- Identify "Loading in %" columns in loading_df (second row) ---
loading_cols = [
    j for j in range(loading_df.shape[1])
    if (
        str(loading_df.iat[1, j]).strip() == 'Loading in %'
        or (
            str(loading_df.iat[0, j]).strip() in ['TSxxx-TR-TRx', 'TSxxx-TR-TRx', 'TSxxx-TR-TRx', 'TSxxx-TR-TRx']
            and str(loading_df.iat[1, j]).strip() == 'Maximum Loading in %'
        )
    )
]

print("Loading columns:", loading_cols)

# --- Build mappings once ---
comp_to_rows = defaultdict(list)
for idx, c in enumerate(vertical_df.iloc[1:, 1], start=1):
    comp_to_rows[str(c).strip()].append(idx)

time_to_col = {str(t).strip(): idx for idx, t in enumerate(vertical_df.iloc[0, 2:], start=2)}

# --- Precompute column-to-row matches ---
col_to_vrows = {}
for j in loading_cols:
    comp_name = str(loading_df.iat[0, j]).strip()
    matched_rows = []
    for v_name, rows in comp_to_rows.items():
        if v_name in comp_name:
            matched_rows.extend(rows)
    if matched_rows:
        col_to_vrows[j] = matched_rows

# --- Vectorized assignment ---
# Instead of row-by-row loops, map whole column at once
for j, v_rows in col_to_vrows.items():
    comp_name = str(loading_df.iat[0, j]).strip()
    print(f"Processing loading column {j} ({comp_name}) with {len(v_rows)} vertical rows...")
    series = loading_df.iloc[:, j]  # all values in this loading column
    time_index = loading_df.iloc[:, 0].astype(str).str.strip()  # all time steps

    # Only update rows that exist in vertical_df
    for t, col_idx in time_to_col.items():
        mask = (time_index == t)
        if mask.any():
            values = series[mask].values
            if len(values) == 1:  # typically one value per time
                for r in v_rows:
                    vertical_df.iat[r, col_idx] = values[0]
            else:
                # If multiple rows per time exist, you may need to align differently
                for r in v_rows:
                    vertical_df.iloc[r, col_idx] = values

# --- Save output ---
vertical_df.to_csv(output_file, index=False, header=False, sep=';')
print("Updated vertical_df saved to:", output_file)
# %%
