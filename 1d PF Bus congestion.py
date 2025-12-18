#%%
import pandas as pd
import numpy as np
from collections import defaultdict

#%%
output_file = r"\\Thesis code\node_congestion_vertical_filled.csv"

# Load the updated vertical_df after filling
df = pd.read_csv(output_file, header=None, sep=';', dtype=str)

timesteps = df.iloc[0, 2:].tolist()

# Convert timestep values to numeric, replace commas with dots if needed
for col in range(2, df.shape[1]):
    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

results = []  # store [parent, max per timestep]

i = 0  # start from first row
while i < len(df):
    parent = df.iat[i, 0]  # parent name in column 0
    if parent and not pd.isna(parent):  # check for non-empty parent
        children_values = []

        j = i + 1
        # Collect children rows until the next parent or empty row
        while j < len(df) and (not df.iat[j, 0] or pd.isna(df.iat[j, 0])):
            if df.iat[j, 1] and not pd.isna(df.iat[j, 1]):
                children_values.append(df.iloc[j, 2:])  # timestep values
            j += 1

        if children_values:
            children_df = pd.DataFrame(children_values)
            max_per_timestep = children_df.max().tolist()
        else:
            max_per_timestep = [None] * (df.shape[1] - 2)

        results.append([parent] + max_per_timestep)
        if len(results) % 50 == 0:
            print(f"Processed {len(results)} parents, currently at row {i}/{len(df)}")
        i = j  # skip to next parent
    else:
        i += 1

# Build output DataFrame

out_df = pd.DataFrame(results, columns=["Parent"] + timesteps)

# Save to CSV
out_df.to_csv("max_per_parent.csv", sep=';', index=False)

print("✅ Finished! Max per parent per timestep:")
print(out_df)

#%%

