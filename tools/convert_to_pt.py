import os
import json
import torch
import pandas as pd
import numpy as np
import ast
import argparse
from torch_geometric.data import Data
from tqdm import tqdm

# =======================
# Utility: Extract EDA stats & mappings
# =======================
def extract_stats_and_mappings(
    vehicle_csv_path,
    junction_csv_path,
    edge_csv_path
):
    def parse_counts(val):
        try:
            if pd.isna(val) or val == '':
                return {}
            return ast.literal_eval(val)
        except Exception:
            return {}

    feature_stats = {'vehicle': {}, 'junction': {}, 'edge': {}}

    def process_csv(path, group):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            feat = row['feature']
            entry = {}
            if row['type'] == 'numeric':
                entry['mean'] = float(row.get('mean', 0.0))
                entry['std'] = float(row.get('std', 1.0)) if float(row.get('std', 1.0)) > 0 else 1.0
            # Always create mapping for 'length' in vehicles as categorical!
            if (group == 'vehicle' and feat == 'length'):
                counts = parse_counts(row.get('value_counts', ''))
                if counts:
                    vals = sorted(float(k) for k in counts.keys())
                else:
                    vals = sorted(set([
                        float(row['min']),
                        float(row['max']),
                        float(row['25%']),
                        float(row['median']),
                        float(row['75%'])
                    ]))
                entry['mapping'] = {v: i for i, v in enumerate(vals)}
            elif row['type'] == 'categorical':
                counts = parse_counts(row.get('value_counts', ''))
                vals = sorted(counts.keys()) if counts else []
                entry['mapping'] = {v: i for i, v in enumerate(vals)}
            feature_stats[group][feat] = entry

    process_csv(vehicle_csv_path, 'vehicle')
    process_csv(junction_csv_path, 'junction')
    process_csv(edge_csv_path, 'edge')
    return feature_stats

# =======================
# Feature Vector Layout
# =======================
def get_feature_layout(stats):
    n_length = len(stats['vehicle']['length']['mapping'])
    n_zone = len(stats['vehicle']['current_zone']['mapping'])
    n_type = len(stats['junction']['type']['mapping']) if 'type' in stats['junction'] else 0
    layout = {
        'n_length': n_length,
        'n_zone': n_zone,
        'n_type': n_type,
        'vehicle':  [
            'node_type', 'length_oh', 'speed', 'acceleration', 'current_x', 'current_y', 'zone_oh',
            'edge_idx', 'current_position', 'sin_hour', 'cos_hour', 'route_length', 'route_length_left', 'type_oh'
        ],
        'junction': [
            'node_type', 'length_oh', 'speed', 'acceleration', 'current_x', 'current_y', 'zone_oh',
            'edge_idx', 'current_position', 'sin_hour', 'cos_hour', 'route_length', 'route_length_left', 'type_oh'
        ]
    }
    feature_dim = (
        1 + n_length + 4 + n_zone + 1 + 1 + 2 + 2 + n_type
    )
    return layout, feature_dim

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
def process_vehicle_nodes(vehicle_nodes, stats, feature_dim):
    vstats = stats['vehicle']
    length_map = vstats['length']['mapping']
    zone_map = vstats['current_zone']['mapping']
    edge_map = stats['edge']['id']['mapping']

    n_length = len(length_map)
    n_zone = len(zone_map)
    n_type = len(stats['junction']['type']['mapping']) if 'type' in stats['junction'] else 0

    features = []
    route_indices = []
    vehicle_ids = []

    for node in vehicle_nodes:
        feat = []
        feat.append(1)
        length_oh = [0] * n_length
        lval = float(node.get("length", 0.0))
        if lval in length_map:
            length_oh[length_map[lval]] = 1
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
        edge_idx = edge_map.get(node.get("current_edge"), 0)
        feat.append(edge_idx)
        mean = vstats['current_position']['mean']
        std = vstats['current_position']['std']
        curr_pos = (node.get("current_position", 0.0) - mean) / std
        feat.append(curr_pos)
        origin_start_sec = node.get("origin_start_sec", 0.0)
        hour = ((origin_start_sec // 3600) % 24) if origin_start_sec > 0 else 0
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)
        feat.extend([sin_hour, cos_hour])
        for key in ['route_length', 'route_length_left']:
            mean = vstats[key]['mean']
            std = vstats[key]['std']
            value = (node.get(key, 0.0) - mean) / std
            feat.append(value)
        feat.extend([0] * n_type)
        if len(feat) < feature_dim:
            feat += [0] * (feature_dim - len(feat))
        elif len(feat) > feature_dim:
            feat = feat[:feature_dim]
        features.append(feat)
        edge_map = stats['edge']['id']['mapping']
        route_raw = node.get("route", [])
        route_idx_list = [edge_map.get(eid, 0) for eid in route_raw]
        route_indices.append(route_idx_list)
        vehicle_ids.append(node.get("id"))
    return torch.FloatTensor(features), route_indices, vehicle_ids

def process_junction_nodes(junction_nodes, stats, feature_dim):
    jstats = stats['junction']
    zone_map = jstats['zone']['mapping'] if 'zone' in jstats and 'mapping' in jstats['zone'] else {}
    type_map = jstats['type']['mapping'] if 'type' in jstats and 'mapping' in jstats['type'] else {}

    n_length = len(stats['vehicle']['length']['mapping'])
    n_zone = len(zone_map)
    n_type = len(type_map)

    features = []
    junction_ids = []

    for node in junction_nodes:
        feat = []
        feat.append(0)
        feat.extend([0] * n_length)
        feat.extend([0, 0, 0, 0])
        zone_oh = [0] * n_zone
        zval = node.get("zone")
        if zval in zone_map:
            zone_oh[zone_map[zval]] = 1
        feat.extend(zone_oh)
        feat.append(0)
        feat.append(0)
        feat.extend([0, 0])
        feat.extend([0, 0])
        type_oh = [0] * n_type
        tval = node.get("type")
        if tval in type_map:
            type_oh[type_map[tval]] = 1
        feat.extend(type_oh)
        if len(feat) < feature_dim:
            feat += [0] * (feature_dim - len(feat))
        elif len(feat) > feature_dim:
            feat = feat[:feature_dim]
        features.append(feat)
        junction_ids.append(node.get("id"))
    return torch.FloatTensor(features), junction_ids

def process_edge_entities(edge_list, stats, edge_route_counts):
    estats = stats['edge']
    features = []
    edge_ids = []
    for edge in edge_list:
        length = (edge.get("length", 0.0) - estats['length']['mean']) / estats['length']['std'] if 'length' in estats else 0.0
        speed = (edge.get("speed", 0.0) - estats['speed']['mean']) / estats['speed']['std'] if 'speed' in estats else 0.0
        count = edge_route_counts.get(edge.get("id"), 0)
        feat = [length, speed, float(count)]
        edge_ids.append(edge.get("id"))
        features.append(feat)
    return torch.FloatTensor(features), edge_ids

def process_labels(label_list):
    label_map = {d['vehicle_id']: d['eta'] for d in label_list}
    return label_map

# =======================
# Main conversion logic
# =======================
def convert_snapshot(snapshot_path, label_path, out_graph_path, stats, feature_dim):
    with open(snapshot_path, 'r') as f:
        snapshot = json.load(f)
    with open(label_path, 'r') as f:
        labels = json.load(f)

    vehicle_nodes = [n for n in snapshot.get("nodes", []) if n.get("node_type") == 1]
    junction_nodes = [n for n in snapshot.get("nodes", []) if n.get("node_type") == 0]
    edges = snapshot.get("edges", [])

    # Compute edge route counts using route_left!
    edge_route_counts = compute_edge_route_counts(vehicle_nodes)

    v_feats, v_route_indices, vehicle_ids = process_vehicle_nodes(vehicle_nodes, stats, feature_dim)
    j_feats, junction_ids = process_junction_nodes(junction_nodes, stats, feature_dim)
    e_feats, edge_ids = process_edge_entities(edges, stats, edge_route_counts)
    label_map = process_labels(labels)

    # Combine node features
    all_node_feats = torch.cat([v_feats, j_feats], dim=0)
    all_node_ids = vehicle_ids + junction_ids

    node_id_to_idx = {nid: idx for idx, nid in enumerate(all_node_ids)}
    edge_index = []
    for edge in edges:
        src_id = edge.get('from')
        tgt_id = edge.get('to')
        if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
            edge_index.append([node_id_to_idx[src_id], node_id_to_idx[tgt_id]])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attrs = e_feats

    label_tensor = torch.FloatTensor([label_map.get(vid, -1) for vid in vehicle_ids])

    data = Data(
        x=all_node_feats,
        edge_index=edge_index,
        edge_attr=edge_attrs,
        y=label_tensor,
        vehicle_ids=vehicle_ids,
        junction_ids=junction_ids,
        edge_ids=edge_ids,
        vehicle_routes=v_route_indices
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
        help="Folder with feature summary CSVs"
    )
    parser.add_argument(
        "--out_graph_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data_pt",
        help="Output folder for .pt graph files"
    )
    args = parser.parse_args()

    os.makedirs(args.out_graph_folder, exist_ok=True)

    vehicle_csv = os.path.join(args.eda_folder, "vehicle_feature_summary.csv")
    junction_csv = os.path.join(args.eda_folder, "junction_feature_summary.csv")
    edge_csv = os.path.join(args.eda_folder, "edge_feature_summary.csv")
    stats = extract_stats_and_mappings(vehicle_csv, junction_csv, edge_csv)
    layout, feature_dim = get_feature_layout(stats)

    snapshot_files = sorted([f for f in os.listdir(args.snapshots_folder) if f.endswith('.json') and "labels" not in f])

    print(f"Converting {len(snapshot_files)} snapshots to PyG Data objects...")
    for snap_file in tqdm(snapshot_files, desc="Processing snapshots"):
        snapshot_path = os.path.join(args.snapshots_folder, snap_file)
        label_file = snap_file.replace("step_", "labels_")
        label_path = os.path.join(args.labels_folder, label_file)
        out_graph_path = os.path.join(args.out_graph_folder, snap_file.replace(".json", ".pt"))
        convert_snapshot(snapshot_path, label_path, out_graph_path, stats, feature_dim)

if __name__ == "__main__":
    main()
