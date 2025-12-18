#%%
import csv
import random
import os
import glob
import pandas as pd

#%% SYncing might reduce speed, turn that off for a while (before was rounding)
# --- File paths ---
file_path = r"\\Thesis code\max_per_parent_rounded.csv"
loads_combined_path = r"\\Bureaublad\Thesis code\loads_combined.csv"
mv_dir = r"C:\EigenData\Powerfactory\Profielen\Stedin Profielen\mv_loads"
trafo_dir = r"C:\EigenData\Powerfactory\Profielen\Stedin Profielen\trafo_loads"
output_dir = r"\\Bureaublad\Adapted loads"

os.makedirs(output_dir, exist_ok=True)

# --- STEP 1: Load agent -> info mapping ---
agent_to_info = {}
with open(loads_combined_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) >= 2:
            agent_to_info[row[0]] = row[1]
agent_info_dict = {}
processed_agents = []

#  STEP 2: Process each agent ---
with open(file_path, newline='') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)  # skip header row

    for row in reader:
        agent = row[0]
        congestion_percentages = row[1:]  # horizontal congestion values per timestep

        # --- Find corresponding profile ---
        info = agent_to_info.get(agent)
        if not info:
            print(f"⚠️ No info found for agent {agent}")
            continue

        profile_file = None
        mv_match = glob.glob(os.path.join(mv_dir, f"*{info}*"))
        trafo_match = glob.glob(os.path.join(trafo_dir, f"*{info}*"))

        if mv_match:
            profile_file = mv_match[0]
        elif trafo_match:
            profile_file = trafo_match[0]

        if not profile_file:
            print(f"❌ No matching file found for agent {agent} ({info})")
            continue

        print(f"✅ Agent {agent} → using profile: {profile_file}")
        agent_info_dict[agent] = {
            "agent": agent,
            "profile_file": profile_file
        }

        # --- Step 3: Read original profile ---
        original_profile_rows = []
        with open(profile_file, newline='') as pf:
            reader_pf = csv.reader(pf, delimiter=';')
            for row_pf in reader_pf:
                if len(row_pf) < 2:
                    continue
                timestep = row_pf[0]
                value_str = row_pf[1].replace(',', '.')  # convert comma to dot
                try:
                    value = float(value_str)
                except ValueError:
                    print(f"⚠️ Invalid value at timestep {timestep}: '{value_str}' → setting to 0")
                    value = 0
                original_profile_rows.append([timestep, value])

        # STEP 4: Compute IncentivizedBehavior resulting flexibility reaction per timestep ---
        merged_rows = []
        prev_reaction = [0,0] # so take two rounds before in account
        duck_curve_multipliers = [0.821, 0.831, 0.846, 0.862, 0.872, 0.898, 0.923, 0.898, 0.821, 0.692, 0.615, 0.564, 0.538, 0.538, 0.564, 0.641, 0.744, 0.872, 0.948, 1.000, 0.975, 0.898, 0.846, 0.810]
        Asset_capacity = round(0.1 + random.betavariate(2, 5) * (0.4), 2) # =R0     ---zodat eenmalig geïnitieerd en niet elke tijdstap
        R2 = round(random.uniform(0, 0.8), 2) #information on capability here?
        for i, (timestep, original_value) in enumerate(original_profile_rows):
            hour_index = i % 24
            R31 =duck_curve_multipliers[hour_index]
            if i < len(congestion_percentages):
                cp = congestion_percentages[i]
                if cp == '' or pd.isna(cp):
                    congestion_percentage = 0
                else:
                    congestion_percentage = float(cp)
                congestion_percentage = int(congestion_percentage)
            else:
                congestion_percentage = 0  # fallback if fewer congestion values than timesteps

            # HERE IS THE CALCULATION OF THE ACTUAL FLEXIBILITY REACTION
            R11 = round(0.3 + random.betavariate(5, 2) * (0.8 - 0.3), 2) #previous round
            R12 = round(random.betavariate(5, 2) * 0.3, 2) #round before that one
            Flex_memory = prev_reaction[0] * R11 + prev_reaction[1] * R12 #takes two round before into account, first round matters more
            inter_asset_cap = Asset_capacity - Flex_memory
            R1 = round(random.betavariate(2, 6) * 0.4, 2)
            Externalities_capacity_loss = inter_asset_cap * R1
            Physical_capacity = Asset_capacity - Flex_memory - Externalities_capacity_loss

            if congestion_percentage > 80:
                R21 = round(random.uniform(0.95, 1.05), 2)
                Externalities_capability_loss = Physical_capacity * R2 * R21
                Capable_physical_capacity = Physical_capacity - Externalities_capability_loss

 
                R3 = round(random.betavariate(9, 2.5), 2) * 0.3 #CMM uncertaintydecision, change in scenarios
                Incentivized_capable_physical_capacity_loss = Capable_physical_capacity * R3 * R31
                Incentivized_capable_physical_capacity = Capable_physical_capacity - Incentivized_capable_physical_capacity_loss

                R4 = round(random.betavariate(1.5, 3) * 0.2, 2)
                Externalities_behavior_loss = Incentivized_capable_physical_capacity * R4
                Incentivized_behavior_capable_physical_capacity = Incentivized_capable_physical_capacity - Externalities_behavior_loss

                output_congestion = congestion_percentage
            else:
                Incentivized_behavior_capable_physical_capacity = 0
                output_congestion = 0
            prev_reaction = [Incentivized_behavior_capable_physical_capacity, prev_reaction[0]]

            merged_rows.append([timestep, original_value, Incentivized_behavior_capable_physical_capacity, output_congestion])
  

        # STEP 5: Save merged results ---
        profile_filename = os.path.basename(profile_file)
        output_path = os.path.join(output_dir, profile_filename)

        with open(output_path, "w", newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["Timestep", "OriginalProfile", "IncentivizedBehavior", "CongestionPercentage"])
            writer.writerows(merged_rows)

        print(f"💾 Saved merged results for {agent} → {output_path}")

        processed_agents.append(agent)

#%% STEP 6 clustering - It loads the edges, creates a mapping, and produces a neighbour list for later usage
# Load only the first two columns of the CSV
edges_df = pd.read_csv(
    r"\\Thesis code\edges_updated.csv",
    usecols=[0, 1],  # Only take columns 0 and 1
    header=None
)
edges_df.columns = ['agent', 'neighbor']

# Remove reversed duplicates to make connections one-way
edges_df['sorted_pair'] = edges_df.apply(lambda row: tuple(sorted([row['agent'], row['neighbor']])), axis=1)
edges_df = edges_df.drop_duplicates(subset='sorted_pair')
edges_df = edges_df.drop(columns='sorted_pair')

# --- CHANGED: ensure all IDs are strings to avoid KeyErrors ---
processed_agents = [str(a) for a in processed_agents]
edges_df['agent'] = edges_df['agent'].astype(str)
edges_df['neighbor'] = edges_df['neighbor'].astype(str)
agent_info_dict = {str(k): v for k, v in agent_info_dict.items()}  # <- CHANGED

agent_connections = {}

for agent in processed_agents:
    if agent not in agent_info_dict:  # <-- CHANGED: skip agents without info
        print(f"Agent {agent} not found in agent_info_dict, skipping.")
        continue

    matches = edges_df[edges_df['agent'] == agent]
    if not matches.empty:
        # Create a list of (neighbor, random_weight) tuples
        neighbors_with_weights = [(neighbor, round(random.random(), 2)) for neighbor in matches['neighbor']]
        agent_connections[agent] = neighbors_with_weights
    else:
        print(f"No connections found for {agent}")

# Print the results
for agent, neighbors in agent_connections.items():
    print(f"{agent} connections with weights: {neighbors}")

print(agent_connections)

agent_neighbor_list = list(
    edges_df[edges_df['agent'].isin(agent_connections.keys())][['agent', 'neighbor']].itertuples(index=False, name=None)
)
print(agent_neighbor_list)

# %% STEP 7 --- This creates the list of file pairs so we can make a tracking of what loads influence each other, but also in what order they should be calculated
new_base = r"\\Bureaublad\Adapted loads"

file_pairs = []

for agent, neighbor in agent_neighbor_list:
    agent_file = agent_info_dict.get(agent, {}).get('profile_file')
    neighbor_file = agent_info_dict.get(neighbor, {}).get('profile_file')
    
    if agent_file and neighbor_file:
        # Replace base path for agent
        agent_filename = agent_file.split('\\')[-1]  # Keep only the filename
        agent_file_new = f"{new_base}\\{agent_filename}"
        
        # Replace base path for neighbor
        neighbor_filename = neighbor_file.split('\\')[-1]
        neighbor_file_new = f"{new_base}\\{neighbor_filename}"
        
        file_pairs.append((agent_file_new, neighbor_file_new))
    else:
        print(f"Warning: Missing profile file for agent {agent} or neighbor {neighbor}")

# Print the resulting list of file pairs
for pair in file_pairs:
    print(pair)


#%% STEP 8 ##CODE below is for ordening agents so agents get adapted first, then neighbours
import os
import re

def extract_numbers(path):
    """Extract all numeric parts from a filename."""
    base = os.path.basename(path)
    return re.findall(r'\d+', base)

def has_overlap(pair1, pair2):
    """Check if two tuples share any numeric ID."""
    nums1 = set(sum([extract_numbers(p) for p in pair1], []))
    nums2 = set(sum([extract_numbers(p) for p in pair2], []))
    return not nums1.isdisjoint(nums2)

def reorder_file_pairs(file_pairs):
    """Reorder file_pairs so similar tuples are adjacent."""
    ordered = []
    remaining = file_pairs.copy()

    while remaining:
        current = remaining.pop(0)
        ordered.append(current)

        # Find next tuple that shares something with the current one
        for i, candidate in enumerate(remaining):
            if has_overlap(current, candidate):
                # Move that tuple right after the current one
                next_pair = remaining.pop(i)
                remaining.insert(0, next_pair)
                break  # continue chaining similarity

    return ordered

file_pairs = reorder_file_pairs(file_pairs)

# ✅ Print final reordered list vertically
print("\nFinal reordered file pairs:\n")
for pair in file_pairs:
    print(pair)

#Run till here for save of 
# %% STEP 9 ---    The neighbour adapts its load in the fifth column thereiwth uses the fifth column and if that is not there uses the third column
for agent_file, neighbor_file in file_pairs:
    try:
        # Load CSVs with comma separator
        df_agent = pd.read_csv(agent_file, sep=',')
        df_neighbor = pd.read_csv(neighbor_file, sep=',')

        # Ensure at least 5 columns in both CSVs
        for df in [df_agent, df_neighbor]:
            while df.shape[1] < 5:
                df.insert(df.shape[1], f'col{df.shape[1]+1}', pd.NA)

        # Determine which column to use from agent: fifth if exists, else third
        agent_col_source = df_agent.iloc[1:, 4]  # fifth column
        # If fifth column is all NA or empty, fall back to third column
        if agent_col_source.isna().all() or agent_col_source.eq('').all():
            agent_col_source = df_agent.iloc[1:, 2]  # third column

        # Convert to numeric, fill NAs with 0
        agent_col = pd.to_numeric(agent_col_source, errors='coerce').fillna(0)
        neighbor_col = pd.to_numeric(df_neighbor.iloc[1:, 2], errors='coerce').fillna(0)

        # Align lengths
        min_len = min(len(agent_col), len(neighbor_col))
        agent_col = agent_col[:min_len]
        neighbor_col = neighbor_col[:min_len]

        # Random weight
        w = round( random.betavariate(1.5, 3), 2)# varying this in scenarios
        
        # Weighted sum
        weighted_sum = agent_col * w + neighbor_col * (1 - w)

        # Overwrite fifth column in neighbor CSV
        df_neighbor.iloc[1:1+min_len, 4] = weighted_sum

        # Save CSV back with comma separator
        df_neighbor.to_csv(neighbor_file, index=False, sep=',')
        print(f"Processed and updated: {neighbor_file}")

    except Exception as e:
        print(f"Error processing pair {agent_file}, {neighbor_file}: {e}")

# %%
