import torch
from torch_geometric.data import Data
import json
import os
import random

def extract_nodes(node_list):
    """
    Converts list of node dictionaries to:
    - node_id_map: mapping node_id -> index
    - node_features: torch tensor for all nodes (vehicles and junctions)
    - node_types: torch tensor for node types (vehicle/junction)
    - node_positions: torch tensor of node coordinates (for visualization)
    """

    node_id_map = {}
    node_features = []
    node_types = []
    node_positions = []

    # Build a sample vehicle feature list to get the correct length
    sample_vehicle_feats = [
        1,  # is_vehicle (flag)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # speed, acceleration, length, width, height, current_x, current_y
        0.0, 0.0, 0.0,                       # current zone one-hot
        0.0, 0.0, 0.0,                       # origin zone one-hot
        0.0, 0.0, 0.0,                       # destination zone one-hot
        0.0, 0.0, 0.0,                       # vehicle type one-hot
        0.0,                                 # route_length
        0.0,                                 # percent route left
        0.0,                                 # origin_start_sec
        0.0, 0.0, 0.0,                       # origin_position, origin_x, origin_y
        0.0, 0.0, 0.0,                       # destination_position, destination_x, destination_y
    ]
    NUM_FEATURES = len(sample_vehicle_feats)

    for idx, node in enumerate(node_list):
        node_id_map[node['id']] = idx

        if node['node_type'] == 1:  # Vehicle
            feats = [
                1,  # is_vehicle (flag)
                node.get('speed', 0.0),
                node.get('acceleration', 0.0),
                node.get('length', 0.0),
                node.get('width', 0.0),
                node.get('height', 0.0),
                node.get('current_x', 0.0),
                node.get('current_y', 0.0),
                # Current zone one-hot
                float(node.get('current_zone', 'A') == 'A'),
                float(node.get('current_zone', 'B') == 'B'),
                float(node.get('current_zone', 'C') == 'C'),
                # Origin zone one-hot
                float(node.get('origin_zone', 'A') == 'A'),
                float(node.get('origin_zone', 'B') == 'B'),
                float(node.get('origin_zone', 'C') == 'C'),
                # Destination zone one-hot
                float(node.get('destination_zone', 'A') == 'A'),
                float(node.get('destination_zone', 'B') == 'B'),
                float(node.get('destination_zone', 'C') == 'C'),
                # Vehicle type one-hot
                float(node.get('vehicle_type', 'passenger') == 'passenger'),
                float(node.get('vehicle_type', '') == 'truck'),
                float(node.get('vehicle_type', '') == 'bus'),
                # Trip info
                node.get('route_length', 0.0),  # total trip distance (meters)
                (node.get('route_length_left', 0.0) / node.get('route_length', 1.0)) if node.get('route_length', 1.0) > 0 else 0.0,  # percent of route left
                node.get('origin_start_sec', 0.0),
                node.get('origin_position', 0.0),
                float(node.get('origin_x', 0.0) or 0.0),
                float(node.get('origin_y', 0.0) or 0.0),
                node.get('destination_position', 0.0),
                float(node.get('destination_x', 0.0) or 0.0),
                float(node.get('destination_y', 0.0) or 0.0),
            ]
        else:  # Junction
            feats = [
                0,  # is_vehicle (flag)
                float(node.get('zone', 'A') == 'A'),
                float(node.get('zone', 'B') == 'B'),
                float(node.get('zone', 'C') == 'C'),
                # Junction type one-hot (only traffic_light and priority)
                float(node.get('type', '') == 'traffic_light'),
                float(node.get('type', '') == 'priority'),
                len(node.get('incoming', [])),
                len(node.get('outgoing', [])),
                node.get('x', 0.0),
                node.get('y', 0.0),
            ]
            # Pad the junction features with zeros to match NUM_FEATURES
            while len(feats) < NUM_FEATURES:
                feats.append(0.0)

        # Optional: debug assertion
        assert len(feats) == NUM_FEATURES, f"Feature length mismatch: {len(feats)} != {NUM_FEATURES}"

        if node['node_type'] == 1:
            x = node.get('current_x', 0.0)
            y = node.get('current_y', 0.0)
        else:
            x = node.get('x', 0.0)
            y = node.get('y', 0.0)

        node_features.append(feats)
        node_types.append(node['node_type'])
        node_positions.append([x, y])

    node_features = torch.tensor(node_features, dtype=torch.float)
    node_types = torch.tensor(node_types, dtype=torch.long)
    node_positions = torch.tensor(node_positions, dtype=torch.float)

    return node_id_map, node_features, node_types, node_positions

def extract_edges(node_list, edge_list, node_id_map):
    """
    Builds edge_index and edge_attr (edge features) for PyTorch Geometric.
    - node_list: list of node dicts
    - edge_list: list of edge dicts (from snapshot['edges'])
    - node_id_map: mapping from node_id to integer index
    Returns: (edge_index, edge_attr)
    """

    edge_index = []
    edge_attr = []

    # Step 1: Calculate future vehicle count for each edge
    future_vehicle_counts = {}
    vehicle_nodes = [node for node in node_list if node['node_type'] == 1]
    for edge in edge_list:
        eid = edge['id']
        future_vehicle_counts[eid] = sum(
            1 for veh in vehicle_nodes if eid in veh.get('route_left', [])
        )

    # Step 2: Build edge features
    for edge in edge_list:
        src_id = edge['from']
        tgt_id = edge['to']
        if src_id not in node_id_map or tgt_id not in node_id_map:
            continue
        edge_index.append([node_id_map[src_id], node_id_map[tgt_id]])
        # One-hot encode zone if present (A, B, C)
        zone = edge.get('zone', 'A')
        zone_A = float(zone == 'A')
        zone_B = float(zone == 'B')
        zone_C = float(zone == 'C')
        # Edge features: [density, avg_speed, future_vehicle_count, num_lanes, length, zone_A, zone_B, zone_C]
        edge_feat = [
            edge.get('density', 0.0),
            edge.get('average_speed', 0.0),
            future_vehicle_counts.get(edge['id'], 0.0),
            edge.get('num_lanes', 1.0),      # default to 1 lane if missing
            edge.get('length', 0.0),         # meters
            zone_A,
            zone_B,
            zone_C
        ]
        edge_attr.append(edge_feat)

    # Convert to PyTorch tensors
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 8), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return edge_index, edge_attr

def load_and_inspect_snapshot(json_path):
    """Load a snapshot JSON file and print a brief summary."""
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Keys in snapshot: {list(data.keys())}")
    print(f"Number of nodes: {len(data['nodes'])}")
    print("First node example:")
    print(data['nodes'][0])  # Show first node for schema inspection

    return data  # So you can use it in other functions

def convert_snapshot_to_pyg(snapshot):
    """
    Converts a single snapshot (dict) to a PyTorch Geometric Data object.
    """
    # Extract nodes and node features
    node_id_map, node_features, node_types, node_positions = extract_nodes(snapshot['nodes'])

    # Extract edges and edge features
    edge_index, edge_attr = extract_edges(snapshot['nodes'], snapshot['edges'], node_id_map)

    # Package into a PyTorch Geometric Data object
    pyg_data = Data(
        x=node_features,          # Node features [num_nodes, num_node_features]
        edge_index=edge_index,    # [2, num_edges]
        edge_attr=edge_attr,      # [num_edges, num_edge_features]
        pos=node_positions,       # [num_nodes, 2] for visualization
        node_types=node_types     # [num_nodes] 0=junction, 1=vehicle
        # Add more fields as needed
    )

    return pyg_data


def compute_eta(vehicle, snapshot_time, gt_entry):
    """Compute ETA for a single vehicle at this snapshot."""
    veh_id = vehicle['id']
    start_time = vehicle.get('origin_start_sec', None)
    arrival_time = gt_entry.get(veh_id, None)
    if start_time is None or arrival_time is None:
        return float('nan')  # or any flag value
    # "Current elapsed" is (snapshot_time - start_time)
    eta = arrival_time - (snapshot_time - start_time)
    if eta < 0:
        eta = 0.0
    return eta

def build_label_tensor(snapshot, gt_dict):
    """
    Returns a tensor where label[i] = ETA for vehicle i,
    NaN for junction nodes.
    """
    num_nodes = len(snapshot['nodes'])
    etas = torch.full((num_nodes,), float('nan'))  # initialize all to nan
    for idx, node in enumerate(snapshot['nodes']):
        if node['node_type'] == 1:
            veh_id = node['id']
            start_time = node.get('origin_start_sec', None)
            arrival_time = gt_dict.get(veh_id, {}).get('destination_time_sec', None)
            snapshot_time = snapshot['step']
            if start_time is not None and arrival_time is not None:
                eta = arrival_time - snapshot_time
                # (optional: clamp negative values to zero)
                if eta < 0:
                    eta = 0.0
                etas[idx] = eta
    return etas


def validate_snapshots(snapshot_dir, percent=0.1, num_print_vehicles=5, verbose=True):
    """
    Randomly samples and validates a percent of snapshot json/pt/label triplets in a directory.

    Args:
        snapshot_dir (str): Directory with .json, .pt, and _labels.pt files
        percent (float): Fraction of files to validate (e.g. 0.1 = 10%)
        num_print_vehicles (int): How many vehicles per file to print info for
        verbose (bool): Whether to print detailed results

    Returns:
        None (prints validation info)
    """
    snapshot_files = [f for f in os.listdir(snapshot_dir) if f.endswith('.json')]
    snapshot_files.sort()
    num_to_validate = max(1, int(len(snapshot_files) * percent))
    sampled_files = random.sample(snapshot_files, num_to_validate)

    print(f"Validating {len(sampled_files)} of {len(snapshot_files)} snapshots ({100*len(sampled_files)/len(snapshot_files):.1f}%)")

    for fname in sampled_files:
        stem = os.path.splitext(fname)[0]
        pt_path = os.path.join(snapshot_dir, f"{stem}.pt")
        label_path = os.path.join(snapshot_dir, f"{stem}_labels.pt")
        json_path = os.path.join(snapshot_dir, fname)

        # Check files exist
        if not (os.path.exists(pt_path) and os.path.exists(label_path)):
            print(f"[WARNING] Missing .pt or label file for {fname}, skipping.")
            continue

        # Load files
        with open(json_path, "r") as f:
            snapshot = json.load(f)
        pyg_data = torch.load(pt_path)
        eta_tensor = torch.load(label_path)

        # Node count check
        if len(snapshot['nodes']) != pyg_data.x.shape[0]:
            print(f"[ERROR] Node count mismatch in {fname}: JSON {len(snapshot['nodes'])}, PT {pyg_data.x.shape[0]}")
        else:
            if verbose:
                print(f"[OK] {fname}: node count matches ({len(snapshot['nodes'])})")

        # Edge count check
        if 'edges' in snapshot:
            num_edges_json = len(snapshot['edges'])
            num_edges_pt = pyg_data.edge_index.shape[1]
            if num_edges_json != num_edges_pt:
                print(f"    [ERROR] Edge count mismatch: json {num_edges_json}, pt {num_edges_pt}")
            elif verbose:
                print(f"    [OK] edge count matches ({num_edges_json})")

        # Vehicle ETA check
        if verbose:
            count_checked = 0
            for i, node in enumerate(snapshot['nodes']):
                if node['node_type'] == 1:
                    eta = eta_tensor[i].item()
                    print(f"    Vehicle {node['id']} index {i} ETA = {eta}")
                    count_checked += 1
                    if count_checked >= num_print_vehicles:
                        break

    print("Validation complete.")

def check_snapshot_labels(snapshot, gt_data, label_tensor, max_vehicles=10, only_print_errors=False):
    """
    For each vehicle in the snapshot, print:
    - vehicle_id, arrival_time, snapshot_time, computed_eta, label_eta, diff
    If only_print_errors=True, print only if diff > 1e-4
    """
    errors_found = 0
    n_printed = 0
    for idx, node in enumerate(snapshot['nodes']):
        if node['node_type'] == 1:
            veh_id = node['id']
            arrival_time = gt_data.get(veh_id, {}).get('destination_time_sec', None)
            snapshot_time = snapshot['step']
            if arrival_time is not None:
                computed_eta = arrival_time - snapshot_time
                label_eta = label_tensor[idx].item()
                diff = abs(label_eta - computed_eta)
                if only_print_errors and diff < 1e-4:
                    continue
                print(f"veh_id: {veh_id}, arrival: {arrival_time}, snapshot: {snapshot_time}, "
                      f"computed_eta: {computed_eta}, label_eta: {label_eta}, diff: {diff}")
                if diff >= 1e-4:
                    errors_found += 1
            else:
                print(f"veh_id: {veh_id}, arrival_time not found in GT.")
                errors_found += 1
            n_printed += 1
            if n_printed >= max_vehicles and not only_print_errors:
                break
    return errors_found

def check_all_snapshot_labels(snapshot_dir, gt_path, max_snapshots=10, max_vehicles=10, only_print_errors=False):
    """
    For each snapshot in snapshot_dir, compare computed ETA to label_tensor.
    """
    # Load ground truth once
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    # Find all json snapshot files
    files = [f for f in os.listdir(snapshot_dir) if f.endswith('.json')]
    files.sort()
    total_errors = 0
    n_checked = 0
    for fname in files:
        stem = os.path.splitext(fname)[0]
        json_path = os.path.join(snapshot_dir, fname)
        label_path = os.path.join(snapshot_dir, f"{stem}_labels.pt")
        if not os.path.exists(label_path):
            print(f"[WARNING] Missing label file for {fname}, skipping.")
            continue
        with open(json_path, "r") as f:
            snapshot = json.load(f)
        label_tensor = torch.load(label_path)
        print(f"\n=== {fname} ===")
        errors = check_snapshot_labels(
            snapshot, gt_data, label_tensor, max_vehicles=max_vehicles, only_print_errors=only_print_errors
        )
        if errors > 0:
            print(f"  [ERROR] Found {errors} mismatches in {fname}")
        n_checked += 1
        if n_checked >= max_snapshots:
            break
    print(f"\nChecked {n_checked} snapshots. Done.")

import os
import json
import torch

def track_vehicles_over_time(snapshot_dir, gt_path, num_snapshots=10, max_vehicles=10):
    """
    Track vehicles from the first snapshot through num_snapshots.
    Print ETA trajectory and check that ETA does not increase.
    """
    # Load ground truth
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    # Get sorted list of snapshot files
    files = [f for f in os.listdir(snapshot_dir) if f.startswith('step_') and f.endswith('.json')]
    files.sort()
    files = files[:num_snapshots]  # Only check first N snapshots

    # Load the first snapshot and label tensor
    first_json_path = os.path.join(snapshot_dir, files[0])
    first_label_path = os.path.join(snapshot_dir, files[0].replace('.json', '_labels.pt'))
    with open(first_json_path, "r") as f:
        first_snapshot = json.load(f)
    first_labels = torch.load(first_label_path)

    # Build vehicle list from first snapshot (limit to max_vehicles)
    tracked_vehicles = []
    for idx, node in enumerate(first_snapshot['nodes']):
        if node['node_type'] == 1:
            veh_id = node['id']
            eta = first_labels[idx].item()
            tracked_vehicles.append((veh_id, eta))
            if len(tracked_vehicles) >= max_vehicles:
                break

    print(f"Tracking {len(tracked_vehicles)} vehicles across {len(files)} snapshots...\n")

    # For each vehicle, collect its ETA in each snapshot
    vehicle_etas = {veh_id: [] for veh_id, _ in tracked_vehicles}

    for snap_idx, fname in enumerate(files):
        json_path = os.path.join(snapshot_dir, fname)
        label_path = os.path.join(snapshot_dir, fname.replace('.json', '_labels.pt'))
        with open(json_path, "r") as f:
            snapshot = json.load(f)
        labels = torch.load(label_path)
        node_id_map = {node['id']: idx for idx, node in enumerate(snapshot['nodes'])}

        for veh_id in vehicle_etas.keys():
            if veh_id in node_id_map:
                idx = node_id_map[veh_id]
                eta = labels[idx].item()
                vehicle_etas[veh_id].append((snap_idx, eta))
            else:
                vehicle_etas[veh_id].append((snap_idx, None))

    # Validate monotonic decrease of ETA
    for veh_id, traj in vehicle_etas.items():
        print(f"\nVehicle {veh_id}:")
        prev_eta = None
        for snap_idx, eta in traj:
            if eta is not None:
                print(f"  Snapshot {snap_idx}: ETA = {eta}")
                if prev_eta is not None and eta > prev_eta + 1e-4:  # small tolerance
                    print(f"    [WARNING] ETA increased! Previous: {prev_eta}, Now: {eta}")
                prev_eta = eta
            else:
                print(f"  Snapshot {snap_idx}: Vehicle not present")


def main():

    SNAPSHOT_DIR = "/media/guy/StorageVolume/ThesisData"
    GT_PATH = "ground_truth.json"  
    
    # 1. Load ground truth file ONCE
    with open(GT_PATH, "r") as f:
        gt_data = json.load(f)  # {veh_id: {arrival_time: int, ...}, ...}
    # 2. Go over all snapshot files
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.json')]
    files.sort()  # sort for reproducibility
    for fname in files:
        json_path = os.path.join(SNAPSHOT_DIR, fname)
        with open(json_path, 'r') as f:
            snapshot = json.load(f)
        # 3. Convert to PyG
        node_id_map, _, _, _ = extract_nodes(snapshot['nodes'])  
        pyg_data = convert_snapshot_to_pyg(snapshot)
        # 4. Get label tensor (eta per vehicle)
        label_tensor = build_label_tensor(snapshot, gt_data)
        # 5. Save
        stem = os.path.splitext(fname)[0]
        torch.save(pyg_data, os.path.join(SNAPSHOT_DIR, f"{stem}.pt"))
        torch.save(label_tensor, os.path.join(SNAPSHOT_DIR, f"{stem}_labels.pt"))
        print(f"Processed: {fname}")
    
    validate_snapshots(SNAPSHOT_DIR, percent=0.1)
    
    track_vehicles_over_time(
        snapshot_dir=SNAPSHOT_DIR,
        gt_path=GT_PATH,
        num_snapshots=20,      # How many snapshots to check (adjust as needed)
        max_vehicles=5         # How many vehicles to track
    )
    
    check_all_snapshot_labels(
        snapshot_dir=SNAPSHOT_DIR,
        gt_path=GT_PATH,
        max_snapshots=5,      # Check 5 snapshots
        max_vehicles=10,      # Show up to 10 vehicles per snapshot
        only_print_errors=False
    )
if __name__ == "__main__":
    main()