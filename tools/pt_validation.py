import os
import json
import torch
import argparse
import random
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from torch_geometric.data import Data

# ---- Extraction logic reused from conversion script ----
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

# --- Core validation function ---

def validate_pt_vs_json(stats, pt_path, gt_json_path, snapshot_json_path, feature_dim, n_to_check=3, tolerance=1e-4):
    """
    For each vehicle node in pt file, print a few sample fields:
    - Original value (from snapshot/gt)
    - Preprocessed value (from pt)
    - Valid/Invalid (within tolerance)
    """
    # Load files
    pt_data = torch.load(pt_path)
    with open(gt_json_path, 'r') as f:
        gt = json.load(f)
    with open(snapshot_json_path, 'r') as f:
        snapshot = json.load(f)

    # Stats and mappings
    vstats = stats['vehicle']
    length_map = vstats['length']['mapping']
    zone_map = vstats['current_zone']['mapping']
    edge_map = stats['edge']['id']['mapping']

    # Map vehicle_id -> original node, and GT label (for eta)
    snapshot_vehicles = {n['id']: n for n in snapshot['nodes'] if n.get('node_type') == 1}
    gt_labels = {x['vehicle_id']: x for x in gt}

    # Reverse mapping for vehicles in pt file
    pt_vehicle_ids = pt_data.vehicle_ids if hasattr(pt_data, 'vehicle_ids') else pt_data['vehicle_ids']
    node_feats = pt_data.x if hasattr(pt_data, 'x') else pt_data['node_features']

    print("\n--- Per-vehicle feature validation (showing up to {} fields per vehicle) ---\n".format(n_to_check))
    all_checks = []
    # For each vehicle
    for idx, vid in enumerate(pt_vehicle_ids):
        node = snapshot_vehicles.get(vid)
        gt_label = gt_labels.get(vid)
        if node is None or gt_label is None:
            continue
        x = node_feats[idx]
        checks = []

        # Length (one-hot, check index)
        lval = float(node.get('length', 0.0))
        lidx = length_map.get(lval, None)
        pt_onehot = x[1:1+len(length_map)].cpu().numpy()
        pt_lidx = np.argmax(pt_onehot)
        checks.append(("length", lval, "one-hot idx: %d" % pt_lidx, pt_lidx == lidx))

        # Speed (normalized)
        mean = vstats['speed']['mean']
        std = vstats['speed']['std']
        original = node.get('speed', 0.0)
        pt_val = float(x[1+len(length_map)])  # feature order!
        normalized = (original - mean) / std
        checks.append(("speed", original, pt_val, abs(pt_val - normalized) < tolerance))

        # Acceleration (normalized)
        mean = vstats['acceleration']['mean']
        std = vstats['acceleration']['std']
        original = node.get('acceleration', 0.0)
        pt_val = float(x[2+len(length_map)])
        normalized = (original - mean) / std
        checks.append(("acceleration", original, pt_val, abs(pt_val - normalized) < tolerance))

        # Zone (one-hot)
        zval = node.get('current_zone')
        zidx = zone_map.get(zval, None)
        pt_zoh = x[5+len(length_map):5+len(length_map)+len(zone_map)].cpu().numpy()
        pt_zidx = np.argmax(pt_zoh)
        checks.append(("current_zone", zval, "one-hot idx: %d" % pt_zidx, pt_zidx == zidx))

        # ETA label (from pt_data.y vs GT)
        eta_idx = idx  # pt_data.y is ordered same as vehicle_ids
        pt_eta = float(pt_data.y[eta_idx]) if hasattr(pt_data, 'y') else float(pt_data['y'][eta_idx])
        gt_eta = float(gt_label.get('eta', -1))
        checks.append(("eta", gt_eta, pt_eta, abs(pt_eta - gt_eta) < tolerance))

        print(f"\nVehicle: {vid}")
        for feat, orig, ptval, valid in checks[:n_to_check]:
            print(f"  {feat}: original={orig} | pt={ptval} | {'VALID' if valid else 'INVALID'}")
        all_checks.extend(checks)
    return all_checks

def sample_and_validate(
    stats, pt_folder, gt_folder, snapshot_folder, feature_dim, n_samples=10, tolerance=1e-4
):
    # Find all pt files and sample 10
    pt_files = sorted([f for f in os.listdir(pt_folder) if f.endswith('.pt')])
    if len(pt_files) == 0:
        print("No .pt files found in", pt_folder)
        return
    files_to_check = random.sample(pt_files, min(n_samples, len(pt_files)))
    all_results = []
    for pt_file in tqdm(files_to_check, desc="Validating random .pt files"):
        base = pt_file.replace(".pt", "")
        gt_json = os.path.join(gt_folder, base.replace("step_", "labels_") + ".json")
        snapshot_json = os.path.join(snapshot_folder, base.replace("labels_", "step_") + ".json")
        pt_path = os.path.join(pt_folder, pt_file)
        if not os.path.exists(gt_json):
            print(f"Missing GT for {pt_file}, gt_json={gt_json}")
            continue
        if not os.path.exists(pt_path):
            print(f"Missing .pt file for {pt_file}, pt_path={pt_path}")
            continue        
          
        checks = validate_pt_vs_json(stats, pt_path, gt_json, snapshot_json, feature_dim, tolerance=tolerance)
        all_results.extend(checks)

    # Summary
    total = len(all_results)
    valid = sum(1 for _, _, _, v in all_results if v)
    print("\n=== Validation summary ===")
    print(f"Total checks: {total}")
    print(f"Valid: {valid} ({valid/total*100:.2f}%)")
    print(f"Invalid: {total - valid} ({100 - valid/total*100:.2f}%)")
    if total > 0 and valid < total:
        print("Some fields did not match preprocessing. Investigate INVALID rows above.")

# ==========================
# Main script with argparse
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Validate .pt files against original snapshot/GT JSONs")
    parser.add_argument(
        "--pt_folder",
        type=str,
        default="/home/guy/Projects/Traffic/traffic_data_pt",
        help="Folder with .pt graph files"
    )
    parser.add_argument(
        "--gt_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data/labels",
        help="Folder with GT labels JSON files"
    )
    parser.add_argument(
        "--snapshot_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data",
        help="Folder with snapshot JSON files"
    )
    parser.add_argument(
        "--eda_folder",
        type=str,
        default="eda_exports",
        help="Folder with feature summary CSVs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of .pt files to sample"
    )
    args = parser.parse_args()

    vehicle_csv = os.path.join(args.eda_folder, "vehicle_feature_summary.csv")
    junction_csv = os.path.join(args.eda_folder, "junction_feature_summary.csv")
    edge_csv = os.path.join(args.eda_folder, "edge_feature_summary.csv")
    stats = extract_stats_and_mappings(vehicle_csv, junction_csv, edge_csv)
    # Feature dim can be computed as in previous script
    n_length = len(stats['vehicle']['length']['mapping'])
    n_zone = len(stats['vehicle']['current_zone']['mapping'])
    n_type = len(stats['junction']['type']['mapping']) if 'type' in stats['junction'] else 0
    feature_dim = 1 + n_length + 4 + n_zone + 1 + 1 + 2 + 2 + n_type

    sample_and_validate(
        stats, args.pt_folder, args.gt_folder, args.snapshot_folder, feature_dim, n_samples=args.n_samples
    )

if __name__ == "__main__":
    main()
