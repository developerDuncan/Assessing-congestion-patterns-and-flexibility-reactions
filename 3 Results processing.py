#%%
import os
import pandas as pd
import numpy as np

# Input and output directories
input_dir = r"C:\\Bureaublad\Adapted loads"
output_dir = r"C:\\Finished loads"

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".csv"):
        continue

    filepath = os.path.join(input_dir, filename)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"⚠️ Could not read {filename}: {e}")
        continue

    # Decide which column to use for 'propogated' logic
    if df.shape[1] >= 5:
        fifth_col_name = df.columns[4]
        target_col = 'propogated'
        df = df.rename(columns={fifth_col_name: target_col})
    elif df.shape[1] >= 3:
        target_col = 'IncentivizedBehavior'
    else:
        print(f"⚠️ Skipping {filename} (not enough columns)")
        continue

    # Convert relevant columns to numeric
    
    # Convert relevant columns to numeric
    df['OriginalProfile'] = pd.to_numeric(df['OriginalProfile'], errors='coerce').fillna(0)
    df['CongestionPercentage'] = (
        df['CongestionPercentage']
        .astype(str)
        .str.replace('%', '', regex=False)
        .astype(float)
        .fillna(0)
    )
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)

    # Reference maximum
    ref_max = df['OriginalProfile'].max()

    # --- reqflex ---
    df['reqflex'] = 0.0
    df.loc[df['CongestionPercentage'] > 80, 'reqflex'] = (
        (df['CongestionPercentage'] - 80) / 100
    )

    # --- actflex ---
    if ref_max == 0:
        df['actflex'] = 0.0
    else:
        # Safe division: avoid inf when OriginalProfile == 0
        df['actflex'] = np.where(
            df['OriginalProfile'] == 0,
            0,
            (ref_max / df['OriginalProfile']) * df[target_col]
        )

        # Extra safety: replace any remaining non-finite values
        df['actflex'] = pd.to_numeric(df['actflex'], errors='coerce')
        df['actflex'] = df['actflex'].replace([np.inf, -np.inf], 0).fillna(0)

    # Save to output directory
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)

    print(f"✅ Processed: {filename}")


print("🎉 All CSVs processed successfully (reqflex only >80, else 0).")



# %%
