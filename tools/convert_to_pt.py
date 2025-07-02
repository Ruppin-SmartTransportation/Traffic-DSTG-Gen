import os
import json
import torch
import pandas as pd
import numpy as np
import ast
import argparse
from torch_geometric.data import Data
from tqdm import tqdm
import datetime
import math

# =======================
# Utility: Extract EDA stats & mappings
# =======================
EXPECTED_NODE_FEATURE_DIM = 20  #
EXPECTED_ZONE_CLASSES = 4
EXPECTED_JUNCTION_TYPE_CLASSES = 2
EXPECTED_VEHICLE_LENGTH_CLASSES = 3

'''
node feature layout:

| Index | Feature Name        | Notes                                                   |
| ----- | ------------------- | ------------------------------------------------------- |
| 0     | `node_type`         | 0 = junction    1 = vehicle                             |
| 1-3   | `length_oh`         | Always `[0, 0, 0]` for junctions                        |
| 4     | `speed`             |                                                         |
| 5     | `acceleration`      |                                                         |
| 6     | `current_x`         |                                                         |
| 7     | `current_y`         |                                                         |
| 8-11  | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
| 12    | `current_edge`      |                                                         |
| 13    | `current_position`  |                                                         |
| 14    | `sin_hour`          |                                                         |
| 15    | `cos_hour`          |                                                         |
| 16    | `sin_day`           |                                                         |
| 17    | `cos_day`           |                                                         |
| 18    | `route_length`      |                                                         |
| 19    | `route_length_left` |                                                         |
| 20    | `j_type`              | junction type, "priority", "traffic_light"  |


'''


def extract_stats_and_mappings(
    vehicle_csv_path,
    junction_csv_path,
    edge_csv_path,
    label_csv_path,
    edge_route_counts_csv_path,
    mapping_folder
):
    def parse_counts(val):
        try:
            if pd.isna(val) or val == '':
                return {}
            return ast.literal_eval(val)
        except Exception:
            return {}

    feature_stats = {'vehicle': {}, 'junction': {}, 'edge': {}, 'labels': {}}
    # Load mappings from JSON files
    vehicle_mapping_path = os.path.join(mapping_folder, "vehicle_mapping.json")
    junction_mapping_path = os.path.join(mapping_folder, "junction_mapping.json")
    edge_mapping_path = os.path.join(mapping_folder, "edge_mapping.json")

    # make sure the mapping files exists
    if not os.path.exists(vehicle_mapping_path):
        print(f"Warning: Vehicle mapping file '{vehicle_mapping_path}' not found.")
    else:
        with open(vehicle_mapping_path, 'r') as f:
            vehicle_data_map = json.load(f)
            print(f"Loaded vehicle  mapping: {len(vehicle_data_map)} entries")
    if not os.path.exists(junction_mapping_path):
        print(f"Warning: Junction mapping file '{junction_mapping_path}' not found.")
    else:
        with open(junction_mapping_path, 'r') as f:
            junction_data_map = json.load(f)
            print(f"Loaded junction mapping: {len(junction_data_map)} entries")
    if not os.path.exists(edge_mapping_path):
        print(f"Warning: Edge mapping file '{edge_mapping_path}' not found.")
    else:       
        with open(edge_mapping_path, 'r') as f:
            edge_data_map = json.load(f)
            print(f"Loaded edge mapping: {len(edge_data_map)} entries")


    # === LABEL STATS ===
    df = pd.read_csv(label_csv_path)
    for _, row in df.iterrows():
        feat = row['feature']
        entry = {
            'mean': float(row.get('mean', 0.0)),
            'std': float(row.get('std', 1.0)) if float(row.get('std', 1.0)) > 0 else 1.0
        }
        feature_stats['labels'][feat] = entry

    # === PROCESS FEATURE CSVS ===
    def process_csv(path, group, id_mapping = None):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            feat = row['feature']
            if feat in feature_stats[group]:
                print(f"Warning: Overwriting feature '{feat}' in group '{group}'")

            entry = {}

            # Handle numeric features
            if row['type'] == 'numeric':
                entry['mean'] = float(row.get('mean', 0.0))
                entry['std'] = float(row.get('std', 1.0)) if float(row.get('std', 1.0)) > 0 else 1.0
                entry['min'] = float(row.get('min', 0.0))
                entry['max'] = float(row.get('max', 1.0))

            # Special handling for vehicle length (categorical mapping)
            if group == 'vehicle' and feat == 'length':
                counts = parse_counts(row.get('value_counts', ''))
                if counts:
                    vals = sorted(float(k) for k in counts.keys())
                else:
                    vals = sorted(set([
                        float(row.get('min', 0.0)),
                        float(row.get('25%', 0.0)),
                        float(row.get('median', 0.0)),
                        float(row.get('75%', 0.0)),
                        float(row.get('max', 0.0))
                    ]))
                entry['mapping'] = {v: i for i, v in enumerate(vals)}

            # Handle categorical features
            elif row['type'] == 'categorical':
                counts = parse_counts(row.get('value_counts', ''))
                vals = sorted(counts.keys()) if counts else []
                entry['mapping'] = {v: i for i, v in enumerate(vals)}
            feature_stats[group][feat] = entry
        # Handle ID mappings
        if id_mapping is not None:
            ids = sorted(id_mapping.keys())
            if group == 'junction':
                vehicle_offset = len(feature_stats['vehicle']) if 'vehicle' in feature_stats else 0
                if vehicle_offset == 0:
                    print("Warning: No vehicle features found, using empty mapping.")
                    exit(1)
                feature_stats[group]['id'] = {
                    'mapping': {v: i+vehicle_offset for i, v in enumerate(ids)}
                }
            else:
                feature_stats[group]['id'] = {
                    'mapping': {v: i for i, v in enumerate(ids)}
                }

    # Process all groups
    process_csv(vehicle_csv_path, 'vehicle', vehicle_data_map)
    process_csv(junction_csv_path, 'junction', junction_data_map)
    process_csv(edge_csv_path, 'edge', edge_data_map)
    process_csv(edge_route_counts_csv_path, 'edge')  # May overwrite some edge features

    # === SUMMARY ===
    print("Feature stats extracted:")
    for group, feats in feature_stats.items():
        print(f"{group}: {len(feats)} features")
        # print one of each feature
        for feat, entry in feats.items():
            if 'mapping' in entry:
                print(f"  {feat}: {len(entry['mapping'])} unique values")
            else:
                print(f"  {feat}: mean={entry.get('mean', 0.0):.4f}, std={entry.get('std', 1.0):.4f}")
        # Uncomment to print all features
        # print(f"  {feat}: {entry}")
        # print(f"  {feat}: {entry.get('mean', 0.0):.4f}, std={entry.get('std', 1.0):.4f}")
        # print(f"  {feat}: {entry.get('min', 0.0):.4f}, max={entry.get('max', 1.0):.4f}")
        # print(f"  {feat}: {entry.get('mapping', {})}")  # Uncomment to print all mappings
        # print(f"  {feat}: {entry.get('value_counts', {})}")  # Uncomment to print all value counts



    return feature_stats

# =======================
# Compute edge route demand
# =======================
def compute_edge_route_counts(vehicle_nodes):
    """
    Returns a dict mapping edge_id -> number of vehicles whose route_left includes edge_id.
    """
    edge_route_counts = {}
    for node in vehicle_nodes:
        route_left = node.get("route_left", [])
        for eid in route_left:
            edge_route_counts[eid] = edge_route_counts.get(eid, 0) + 1
    return edge_route_counts

# =======================
# Processing Functions
# =======================
def process_vehicle_nodes(vehicle_nodes, stats):
    vstats = stats['vehicle']
    features = []
    curr_route_indices = []
    curr_vehicle_ids = []
    vehicle_length_map = {4.5: 0, 8.0: 1, 12.0: 2}
    n_length = len(vehicle_length_map)
    edge_map = stats['edge']['id']['mapping'] if 'edge' in stats and 'id' in stats['edge'] and 'mapping' in stats['edge']['id'] else {}
    if edge_map is None or len(edge_map) == 0:
        print("Warning: Edge mapping is empty or not found, using default empty mapping.")
        exit(1)

    global zone_map, n_zone

    for node in vehicle_nodes:
        feat = []
        feat.append(1)
        length_oh = [0] * n_length
        lval = float(node.get("length", 0.0))
        if lval in vehicle_length_map:
            length_oh[vehicle_length_map[lval]] = 1
        feat.extend(length_oh)
        for key in ['speed', 'acceleration', 'current_x', 'current_y']:
            mean = vstats[key]['mean']
            std = vstats[key]['std']
            value = (node.get(key, 0.0) - mean) / std
            feat.append(value)
        zone_oh = [0] * n_zone
        zval = node.get("current_zone")
        if zval in zone_map:
            zone_oh[zone_map[zval]] = 1
        feat.extend(zone_oh)
        # edge_idx = edge_map.get(node.get("current_edge"), -1)
        # if edge_idx == -1:
        #     print(f"Warning: Edge {node.get('current_edge')} not found in edge mapping, using index 0.")
        # feat.append(edge_idx)
        mean = vstats['current_position']['mean']
        std = vstats['current_position']['std']
        curr_pos = (node.get("current_position", 0.0) - mean) / std
        feat.append(curr_pos)
        origin_start_sec = node.get("origin_start_sec", 0.0)

        try:
            dt = datetime.datetime.utcfromtimestamp(origin_start_sec)
            hour = dt.hour + dt.minute / 60.0
            weekday = dt.weekday()  # Monday=0, Sunday=6

            sin_hour = np.sin(2 * np.pi * hour / 24)
            cos_hour = np.cos(2 * np.pi * hour / 24)
            sin_day = np.sin(2 * np.pi * weekday / 7)
            cos_day = np.cos(2 * np.pi * weekday / 7)
        except:
            sin_hour = cos_hour = sin_day = cos_day = 0.0  # Fallback for invalid timestamps

        feat.extend([sin_hour, cos_hour, sin_day, cos_day])

        for key in ['route_length', 'route_length_left']:
            mean = vstats[key]['mean']
            std = vstats[key]['std']
            value = (node.get(key, 0.0) - mean) / std
            feat.append(value)

        if len(feat) < EXPECTED_NODE_FEATURE_DIM:
            feat += [0] * (EXPECTED_NODE_FEATURE_DIM - len(feat))
            
        features.append(feat)
        
        route_raw = node.get("route", [])
        route_idx_list = [edge_map.get(eid, -1) for eid in route_raw]
        if -1 in route_idx_list:
            print(f"Warning: Some edges in route {route_raw} not found in edge mapping.")
            # print the missing edges
            missing_edges = [eid for eid in route_raw if eid not in edge_map]
            print(f"Missing edges: {missing_edges}")
            exit(1)
        curr_route_indices.append(route_idx_list)
        curr_vehicle_ids.append(node.get("id"))
    return torch.FloatTensor(features), curr_route_indices, curr_vehicle_ids

def process_junction_nodes(junction_nodes, stats):
    
    features = []
    for node in junction_nodes:
        feat = []
        feat.extend([0] * 6)
        jstats = stats['junction']
        for key in ['x', 'y']:
            mean = jstats[key]['mean']
            std = jstats[key]['std']
            value = (node.get(key, 0.0) - mean) / std
            feat.append(value)

        zone_oh = [0] * n_zone
        zval = node.get("zone")
        if zval in zone_map:
            zone_oh[zone_map[zval]] = 1
        feat.extend(zone_oh)
        feat.extend([0] * 7)

        # Junction type
        jtype = node.get("type", "priority")
        feat.append(1 if jtype == "traffic_light" else 0)

        features.append(feat)
    return torch.FloatTensor(features)

def process_edge_entities(edge_list, stats, edge_route_counts):
    estats = stats['edge']
    max_lanes = 3  # One-hot range for num_lanes ∈ {1,2,3}
    features = []
    edge_ids = []

    for edge in edge_list:
        # 1. Normalized avg_speed
        avg_speed = edge.get("avg_speed", 0.0)
        if 'avg_speed' in estats:
            mean = estats['avg_speed']['mean']
            std = estats['avg_speed']['std'] or 1.0
            avg_speed = (avg_speed - mean) / std
        else:
            avg_speed = 0.0

        # 2. One-hot encode num_lanes
        num_lanes = int(edge.get("num_lanes", 1))
        one_hot_lanes = [1 if i == num_lanes else 0 for i in range(1, max_lanes + 1)]

        # 3. vehicles_on_road_count → log + z-score normalization
        vlist = edge.get("vehicles_on_road", [])
        vcount = len(vlist)
        vcount_log = math.log1p(vcount)
        if 'vehicles_on_road_count' in estats:
            mean = estats['vehicles_on_road_count_log']['mean']
            std = estats['vehicles_on_road_count_log']['std'] or 1.0
            vcount_norm = (vcount_log - mean) / std
        else:
            print("Warning: 'vehicles_on_road_count_log' stats not found, using default normalization.")
            vcount_norm = 0.0

        # 4. edge_route_count → log + z-score normalization capped to clip_max
        route_count = edge_route_counts.get(edge.get("id"), 0)
        route_count_log = math.log1p(route_count)
        if 'edge_route_count_log' in estats:
            mean = estats['edge_route_count_log']['mean']
            std = estats['edge_route_count_log']['std'] or 1.0
            route_count_norm = (route_count_log - mean) / std
        else:
            print("Warning: 'edge_route_count_log' stats not found, using default normalization.")
            route_count_norm = 0.0

        # Compose final feature vector
        feat = [avg_speed] + one_hot_lanes + [vcount_norm, route_count_norm]
        edge_ids.append(edge.get("id"))
        features.append(feat)

    return torch.FloatTensor(features), edge_ids

def process_labels(label_list, stats, normalize_labels, filter_outliers):
    """
    Processes label data into a map and filters out vehicle IDs based on ETA outliers.

    Args:
        label_list (list): List of label dictionaries from a single snapshot.
        stats (dict): Parsed stats including 'labels' from labels_feature_summary.csv.
        normalize_labels (bool): If True, apply z-score normalization to ETA.
        filter_outliers (bool): If True, remove labels with |z-score| > 3.

    Returns:
        label_map (dict): {vehicle_id: eta_value}
        valid_vehicle_ids (set): Set of vehicle IDs to keep
    """
    label_map = {}
    valid_vehicle_ids = set()

    eta_stats = stats.get("labels", {}).get("eta", {})
    mean_eta = eta_stats.get("mean", 0.0)
    std_eta = eta_stats.get("std", 1.0)
    if std_eta == 0.0:
        std_eta = 1.0
    filter_outliers_count = 0
    for entry in label_list:
        eta = entry.get('eta')
        vid = entry.get('vehicle_id')
        if eta is None or vid is None:
            continue

        if normalize_labels or filter_outliers:
            z = (eta - mean_eta) / std_eta

            if filter_outliers and abs(z) > 3:
                filter_outliers_count += 1
                continue  # Outlier → skip

            eta_final = z if normalize_labels else eta
        else:
            eta_final = eta

        label_map[vid] = eta_final
        valid_vehicle_ids.add(vid)

    return label_map, valid_vehicle_ids


# =======================
# Main conversion logic
# =======================
def convert_snapshot(snapshot_path, label_path, out_graph_path, stats, normalize_labels, filter_outliers):

    with open(snapshot_path, 'r') as f:
        snapshot = json.load(f)
    with open(label_path, 'r') as f:
        labels = json.load(f)
    
    label_map, valid_vehicle_ids = process_labels(labels, stats, normalize_labels, filter_outliers)
    curr_vehicle_nodes = [n for n in snapshot.get("nodes", []) if n.get("node_type") == 1 and n.get("id") in valid_vehicle_ids]
    junction_nodes = [n for n in snapshot.get("nodes", []) if n.get("node_type") == 0]
    junction_ids = [n.get("id") for n in junction_nodes]
    if len(junction_ids) != len(stats['junction']['id']['mapping'].keys()):
        print(f"Warning: Junction IDs in snapshot ({len(junction_ids)}) do not match stats mapping ({len(stats['junction']['id']['mapping'])}).")
        exit(1)
    edges = snapshot.get("edges", [])

    # Compute edge route counts using route_left!
    curr_edge_route_counts = compute_edge_route_counts(curr_vehicle_nodes)

    v_feats, curr_route_indices, curr_vehicle_ids = process_vehicle_nodes(curr_vehicle_nodes, stats)
    j_feats = process_junction_nodes(junction_nodes, stats)
    e_feats, edge_ids = process_edge_entities(edges, stats, curr_edge_route_counts)
    if len(edge_ids) != len(stats['edge']['id']['mapping'].keys()):
        print(f"Warning: Edge IDs in snapshot ({len(edge_ids)}) do not match stats mapping ({len(stats['edge']['id']['mapping'])}).")
        exit(1)
    # Combine node features
    all_node_feats = torch.cat([v_feats, j_feats], dim=0)
    # Create a mapping from node IDs to indices from stats mapping
    curr_vehicles_id_to_idx = {vid: idx for vid, idx in stats['vehicle']['id']['mapping'].items() if vid in curr_vehicle_ids}
    junctions_id_to_idx = stats['junction']['id']['mapping']

    #append junction_id_to_idx to curr_vehicles_id_to_idx
    nodes_id_to_idx = {**curr_vehicles_id_to_idx, **junctions_id_to_idx}
    nodes_idx_list = list(nodes_id_to_idx.values())

    edge_index = []
    for edge in edges:
        src_id = edge.get('from')
        tgt_id = edge.get('to')
        if src_id in nodes_id_to_idx and tgt_id in nodes_id_to_idx:
            edge_index.append([junctions_id_to_idx[src_id], junctions_id_to_idx[tgt_id]])
        else:
            print(f"Warning: Edge {src_id} -> {tgt_id} has nodes not in mapping, exit.")
            print(f"Missing nodes: {src_id} -> {tgt_id}")
            exit(1)
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        print("Warning: No valid edges found in snapshot.")
        exit(1)
    edge_attrs = e_feats

    label_tensor = torch.FloatTensor([
        label_map.get(vid, -1) for vid in curr_vehicle_ids
    ])
    if len(label_tensor) != len(curr_vehicle_ids):
        print(f"Warning: Label tensor length ({len(label_tensor)}) does not match vehicle IDs length ({len(curr_vehicle_ids)}).")
        exit(1)
    if -1 in label_tensor:
        print("Warning: Some vehicle IDs do not have labels, exiting.")
        print(f"Missing labels for vehicle IDs: {[vid for vid in curr_vehicle_ids if vid not in label_map]}")
        exit(1)

    data = Data(
        x=all_node_feats,
        node_idxs=torch.tensor(nodes_idx_list, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attrs,
        y=label_tensor,
        vehicle_ids=curr_vehicle_ids,
        junction_ids=junction_ids,
        edge_ids=edge_ids,
        vehicle_routes=curr_route_indices
    )
    torch.save(data, out_graph_path)

# =======================
# Main script
# =======================
def main():
    parser = argparse.ArgumentParser(description="Convert traffic snapshots and labels to .pt files for GNN training")
    parser.add_argument(
        "--snapshots_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data",
        help="Folder with snapshot JSON files"
    )
    parser.add_argument(
        "--labels_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data/labels",
        help="Folder with per-snapshot label JSON files"
    )
    parser.add_argument(
        "--eda_folder",
        type=str,
        default="eda_exports",
        help="Folder with labels and feature summary CSVs"
    )
    parser.add_argument(
        "--out_graph_folder",
        type=str,
        default="/home/guy/Projects/Traffic/traffic_data_pt",
        help="Output folder for .pt graph files"
    )
    parser.add_argument(
    "--filter_outliers",
    action="store_true",
    default=True,
    help="Enable filtering of outliers in ETA labels (default: True)"
    )

    parser.add_argument(
        "--normalize_labels",
        action="store_true",
        default=True,
        help="Enable z-score normalization of ETA labels (default: True)"
    )

    parser.add_argument(
        "--mapping_folder",
        action="store_true",
        default="mappings",
        help="Folder with mappings for vehicle, junction, and edge features (default: eda_exports/mappings)"
    )

    args = parser.parse_args()

    os.makedirs(args.out_graph_folder, exist_ok=True)

    vehicle_csv = os.path.join(args.eda_folder, "vehicle_feature_summary.csv")
    junction_csv = os.path.join(args.eda_folder, "junction_feature_summary.csv")
    edge_csv = os.path.join(args.eda_folder, "edge_feature_summary.csv")
    label_csv = os.path.join(args.eda_folder, "labels_feature_summary.csv")
    edge_route_counts_csv = os.path.join(args.eda_folder, "edge_route_count_summary.csv")
    mapping_folder = os.path.join(args.eda_folder, args.mapping_folder)
    stats_and_id_mapping = extract_stats_and_mappings(vehicle_csv, junction_csv, edge_csv, label_csv, edge_route_counts_csv, mapping_folder)
    
    global zone_map, n_zone
    zone_map = {'A': 0, 'B': 1, 'C': 2, 'H':3}
    n_zone = len(zone_map)

    snapshot_files = sorted([f for f in os.listdir(args.snapshots_folder) if f.endswith('.json') and "labels" not in f])

    print(f"Converting {len(snapshot_files)} snapshots to PyG Data objects...")
    for snap_file in tqdm(snapshot_files, desc="Processing snapshots"):
        snapshot_path = os.path.join(args.snapshots_folder, snap_file)
        label_file = snap_file.replace("step_", "labels_")
        label_path = os.path.join(args.labels_folder, label_file)
        out_graph_path = os.path.join(args.out_graph_folder, snap_file.replace(".json", ".pt"))
        convert_snapshot(
            snapshot_path,
            label_path,
            out_graph_path,
            stats_and_id_mapping,
            normalize_labels=args.normalize_labels,
            filter_outliers=args.filter_outliers
        )


if __name__ == "__main__":
    main()
