import os
import json
import random
import torch
import numpy as np
import pandas as pd
import ast

PT_DIR = "/home/guy/Projects/Traffic/traffic_data_pt"
SNAPSHOT_DIR = "/media/guy/StorageVolume/traffic_data"
LABEL_DIR = "/media/guy/StorageVolume/traffic_data/labels"
EDA_DIR = "eda_exports"
MAPPING_DIR = os.path.join(EDA_DIR, "mappings")
ERROR_MARGIN = 1e-3
SAMPLE_SIZE = 1000

# Load statistics
vehicle_stats = pd.read_csv(os.path.join(EDA_DIR, "vehicle_feature_summary.csv"), index_col="feature").to_dict(orient="index")
junction_stats = pd.read_csv(os.path.join(EDA_DIR, "junction_feature_summary.csv"), index_col="feature").to_dict(orient="index")
edge_stats = pd.read_csv(os.path.join(EDA_DIR, "edge_feature_summary.csv"), index_col="feature").to_dict(orient="index")
label_stats = pd.read_csv(os.path.join(EDA_DIR, "labels_feature_summary.csv"), index_col="feature").to_dict(orient="index")

# Load mappings
with open(os.path.join(MAPPING_DIR, "vehicle_mapping.json")) as f:
    vehicle_mapping = json.load(f)
with open(os.path.join(MAPPING_DIR, "junction_mapping.json")) as f:
    junction_mapping = json.load(f)
with open(os.path.join(MAPPING_DIR, "edge_mapping.json")) as f:
    edge_mapping = json.load(f)

# Helper: invert min-max normalization
def invert_minmax(val, minv, maxv):
    return val * (maxv - minv) + minv

# Helper: invert z-score normalization
def invert_zscore(val, mean, std):
    return val * std + mean

# Helper: invert log+z-score normalization
def invert_log_zscore(val, mean, std):
    return np.expm1(val * std + mean)

# Instead of sampling 1000, pick a single random .pt file for detailed comparison
pt_files = [f for f in os.listdir(PT_DIR) if f.endswith('.pt')]
if not pt_files:
    print("No .pt files found.")
    exit(1)
pt_file = random.choice(pt_files)
pt_path = os.path.join(PT_DIR, pt_file)
data = torch.load(pt_path)
step_num = pt_file.replace("step_", "").replace(".pt", "")
snap_file = f"step_{step_num}.json"
label_file = f"labels_{step_num}.json"
snap_path = os.path.join(SNAPSHOT_DIR, snap_file)
label_path = os.path.join(LABEL_DIR, label_file)
if not os.path.exists(snap_path) or not os.path.exists(label_path):
    print(f"Missing snapshot or label for {pt_file}")
    exit(1)
with open(snap_path) as f:
    snap_json = json.load(f)
with open(label_path) as f:
    label_json = json.load(f)
# Build a map from vehicle_id to label entry for fast lookup
label_map = {entry["vehicle_id"]: entry for entry in label_json}
travel_time_99p = label_stats["total_travel_time_seconds"]["99%"]
vehicle_nodes_json = []
vehicle_ids_json = []
for n in snap_json["nodes"]:
    if n.get("node_type") == 1:
        vid = n["id"]
        label_entry = label_map.get(vid)
        if label_entry is None:
            continue
        duration = label_entry.get("total_travel_time_seconds", 0)
        if duration < 180 or duration > travel_time_99p:
            continue
        vehicle_nodes_json.append(n)
        vehicle_ids_json.append(vid)
vehicle_ids_pt = data.vehicle_ids
if vehicle_ids_pt != vehicle_ids_json:
    print(f"Vehicle ID alignment mismatch in {pt_file}")
    print(f"  .pt vehicle_ids (first 10): {vehicle_ids_pt[:10]}")
    print(f"  JSON vehicle_ids (first 10): {vehicle_ids_json[:10]}")
    print(f"  .pt vehicle_ids length: {len(vehicle_ids_pt)}")
    print(f"  JSON vehicle_ids length: {len(vehicle_ids_json)}")
    exit(1)
# Robustly extract vehicle_type keys
veh_type_keys = vehicle_stats["vehicle_type"].get("keys")
if not veh_type_keys or veh_type_keys == '' or veh_type_keys == []:
    value_counts = vehicle_stats["vehicle_type"].get("value_counts")
    if value_counts:
        try:
            veh_type_keys = list(ast.literal_eval(value_counts).keys())
        except Exception:
            veh_type_keys = []
if isinstance(veh_type_keys, str):
    veh_type_keys = ast.literal_eval(veh_type_keys)
if not veh_type_keys:
    print("vehicle_stats['vehicle_type']:", vehicle_stats["vehicle_type"])
    raise RuntimeError("Could not extract vehicle_type keys from vehicle_stats")
# Print comparison for all vehicles in this snapshot
print(f"\nComparison for snapshot {snap_file} and PT file {pt_file}:")
offset = len(data.junction_ids)
for i, vid in enumerate(vehicle_ids_pt):
    x_idx = offset + i
    x_vec = data.x[x_idx].numpy()
    node_json = vehicle_nodes_json[i]
    print(f"\nVehicle {vid}:")
    # node_type
    print(f"  node_type: PT={x_vec[0]}, original=1")
    # veh_type_oh
    veh_type = node_json["vehicle_type"]
    veh_type_oh = [0]*len(veh_type_keys)
    veh_type_oh[veh_type_keys.index(veh_type)] = 1
    print(f"  veh_type_oh: PT={x_vec[1:4].tolist()}, original={veh_type_oh}")
    # speed
    speed = node_json["speed"]
    minv, maxv = vehicle_stats["speed"]["min"], vehicle_stats["speed"]["max"]
    speed_inv = invert_minmax(x_vec[4], minv, maxv)
    print(f"  speed: PT={speed_inv}, original={speed}")
    # acceleration
    acc = node_json["acceleration"]
    minv, maxv = vehicle_stats["acceleration"]["min"], vehicle_stats["acceleration"]["max"]
    acc_inv = invert_minmax(x_vec[5], minv, maxv)
    print(f"  acceleration: PT={acc_inv}, original={acc}")
    # route_length
    rl = node_json["route_length"]
    minv, maxv = vehicle_stats["route_length"]["min"], vehicle_stats["route_length"]["max"]
    rl_inv = invert_minmax(x_vec[10], minv, maxv)
    print(f"  route_length: PT={rl_inv}, original={rl}")
    # route_length_left
    rll = node_json["route_length_left"]
    minv, maxv = vehicle_stats["route_length_left"]["min"], vehicle_stats["route_length_left"]["max"]
    rll_inv = invert_minmax(x_vec[11], minv, maxv)
    print(f"  route_length_left: PT={rll_inv}, original={rll}")
    # current_x
    cx = node_json["current_x"]
    minv, maxv = vehicle_stats["current_x"]["min"], vehicle_stats["current_x"]["max"]
    cx_inv = invert_minmax(x_vec[16], minv, maxv)
    print(f"  current_x: PT={cx_inv}, original={cx}")
    # current_y
    cy = node_json["current_y"]
    minv, maxv = vehicle_stats["current_y"]["min"], vehicle_stats["current_y"]["max"]
    cy_inv = invert_minmax(x_vec[17], minv, maxv)
    print(f"  current_y: PT={cy_inv}, original={cy}")
    # label (ETA)
    label_entry = label_map.get(vid)
    if label_entry:
        eta = label_entry.get("eta")
        print(f"  label (ETA): original={eta}")
print("\nDone.") 