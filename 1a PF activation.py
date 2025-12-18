#%%
# -------------------------------
# IMPORTS
# -------------------------------
import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add PowerFactory Python path
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP4A\Python\3.10")
import powerfactory as pf

# -------------------------------
# CONNECT TO POWERFACTORY
# -------------------------------
app = pf.GetApplication()
if app is None:
    raise Exception("❌ Could not connect to PowerFactory.")

user = app.GetCurrentUser()
print("✅ Connected to PowerFactory as user:", user)
app.Show()  # optional

# -------------------------------
# ACTIVATE PROJECT
# -------------------------------
PROJECT_NAME = "Combi(5)"
app.ActivateProject(PROJECT_NAME)
proj = app.GetActiveProject()
print("✅ Project activated:", proj.loc_name)

# -------------------------------
# GET ACTIVE STUDY CASE
# -------------------------------
study_case = app.GetActiveStudyCase()
if study_case is None:
    raise Exception("❌ No active study case found.")
print("✅ Active study case:", study_case.loc_name)

#%%
network = study_case.GetContents("*.ElmNet")
if not network:
    raise Exception("⚠️ No network found in the active study case!")
network = network[0]  # usually only one network
print("✅ Active network:", network.loc_name)

# List all elements in this network
all_elements = network.GetContents("*")
print(f"✅ {len(all_elements)} elements found in the active network:")
for e in all_elements:
    print("-", e.loc_name, e.GetClassName())

#%%
# Get all lines
lines = app.GetCalcRelevantObjects('*.ElmLne')

# Print the lines
print("Number of lines:", len(lines))
for line in lines:
    print("Line name:", line.loc_name)        # Local name of the line
    print("From bus:", line.bus1.loc_name)    # Sending end bus
    print("To bus:", line.bus2.loc_name)      # Receiving end bus
    print("R (resistance):", line.R1)        # Line resistance
    print("X (reactance):", line.X1)          # Line reactance
    print("----")


#%% PERFORMING THE ACTUAL LOAD FLOW
# Get all objects matching the pattern in the active study case
statsim_list = study_case.GetContents("*.ComStatsim*")
if not statsim_list:
    raise Exception("❌ No Quasi-Dynamic simulation object found")

# Use the first match
statsim = statsim_list[0]
print("✅ Found ComStatsim:", statsim.loc_name)

# Execute the QD simulation
status = statsim.Execute()  # runs the simulation
if status == 0:
    print("✅ Quasi-dynamic simulation completed successfully")
else:
    print(f"❌ Simulation failed with status code: {status}")

#%%
# Lines
lines = app.GetCalcRelevantObjects('*.ElmLne')          # all lines

# Two-winding transformers
trafos2w = app.GetCalcRelevantObjects('*.ElmTr2')      # two-winding transformers

# Three-winding transformers
trafos3w = app.GetCalcRelevantObjects('*.ElmTr3')      # three-winding transformers

#%% Print element names

print("=== Lines ===")
for line in lines:
    print(line.loc_name)
print(f"Total number of lines: {len(lines)}\n")

print("=== Two-Winding Transformers ===")
for trafo in trafos2w:
    print(trafo.loc_name)
print(f"Total number of two-winding transformers: {len(trafos2w)}\n")

print("=== Three-Winding Transformers ===")
for trafo in trafos3w:
    print(trafo.loc_name)
print(f"Total number of three-winding transformers: {len(trafos3w)}\n")

#%%
import csv

loads = app.GetCalcRelevantObjects('*.ElmLod')
loads_mv = app.GetCalcRelevantObjects('*.ElmLodmv')

# Export ElmLod
with open("loads_elmlod.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["loc_name", "substation", "for_name", "chr_name", "class"])

    for load in loads:
        terminal = getattr(load, 'bus1', None)
        cterm = getattr(terminal, 'cterm', None) if terminal else None
        substation = getattr(cterm, 'cpSubstat', None) if cterm else None

        writer.writerow([
            load.loc_name,
            substation.loc_name if substation else "",
            "",  # no for_name for ElmLod
            getattr(load, 'chr_name', ""),
            "ElmLod"
        ])
import re

with open("loads_elmlodmv.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["loc_name", "for_name", "chr_name", "busbar", "class"])  # added busbar column

    for load in loads_mv:
        loc_name = load.loc_name

        # default busbar = ""
        busbar = ""
        # look for pattern LOAD_xxx-TR...
        match = re.search(r"LOAD_(.*?)-TR", loc_name)
        if match:
            busbar = match.group(1)

        writer.writerow([
            loc_name,
            getattr(load, 'for_name', ""),
            getattr(load, 'chr_name', ""),
            busbar,
            "ElmLodmv"
        ])

print(f"✅ Exported {len(loads_mv)} ElmLodmv loads to loads_elmlodmv.csv")


# %%
# Read ElmLod data
with open("loads_elmlod.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    elmlod_data = list(reader)

# Read ElmLodmv data
with open("loads_elmlodmv.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    elmlodmv_data = list(reader)

# Create combined CSV
with open("loads_combined.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["group", "info"])  # group = substation/busbar, info = chr_name-loc_name or chr_name

    # ElmLod entries
    for row in elmlod_data:
        substation = row["substation"]
        chr_name = row["chr_name"]
        loc_name = row["loc_name"]

        writer.writerow([
            substation,
            f"{chr_name}-{loc_name}"
        ])

    # ElmLodmv entries
    for row in elmlodmv_data:
        busbar = row["busbar"]
        chr_name = row["chr_name"]

        writer.writerow([
            busbar,
            f"*{chr_name}"
        ])

print("✅ Exported combined ElmLod + ElmLodmv data to loads_combined.csv")

# %%
