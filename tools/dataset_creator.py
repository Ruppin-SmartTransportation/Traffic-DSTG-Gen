'''
x - Nodes (Junctions, Vehicle) main feature layout:

| Index | Feature Name        | Notes                                                   |
| ----- | ------------------- | ------------------------------------------------------- |
| 0     | `node_type`         | 0 = junction    1 = vehicle                             |
| 1-3   | `length_oh`         | Always `[0, 0, 0]` for junctions                        |
| 4     | `speed`             | z normalized                                            |
| 5     | `acceleration`      | z normalized                                            |
| 6     | `sin_hour`          | represent time in a unit circle                         |                             |
| 7     | `cos_hour`          | represent time in a unit circle                         |                                |
| 8     | `sin_day`           | represent day in a unit circle                          |                              |
| 9     | `cos_day`           | represent day in a unit circle                          |                              |
| 10    | `route_length`      | z normalized                                            |
| 11    | `route_length_left` | z normalized                                            |
| 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
| 16    | `current_x`         | z normalized                                            |
| 17    | `current_y`         | z normalized                                            |
| 18    | `j_type`            | junction type, "priority", "traffic_light"              |

Total of 19 features per node.

edge_features - Roads (static Edges) main feature layout:

| Index | Feature Name        | Notes                                                   |

| ----- | ---------------------- | ---------------------------------------------------- |
| 0     | `avg_speed`            | z normalized                                         |
| 1-3   | `num_lanes`            | One-hot of 1-3 lanes                                 |
| 4     | `edge_demand`          | log + z-score normalized                             |
| 5     | `edge_occupancy`       | z normalized                                         |
| 6     | `length`               | z normalized (distance between from and to nodes)    |

Total of 7 features per edge.

node indexing:

junctions_id_to_idx - {'J1': 0, 'J2': 1, ...}
offset = len(junctions_id_to_idx)  # Offset for vehicle nodes in the node features
vehicles_id_to_idx = {vid: idx + offset for idx, vid in enumerate(curr_vehicle_ids)}
nodes_id_to_idx = {**junction_id_to_idx, **vehicle_id_to_idx}

edge indexing:
edges_id_to_idx = {'AX3AX2': 0, 'AX4AX3': 1, ...} (only static edges)
edge_index = [
    [src_node_0, src_node_1, ..., src_node_N],
    [tgt_node_0, tgt_node_1, ..., tgt_node_N]
]
edge_type [0, 1, 1, 1, 2, 2 ,...] 

| Edge Type Code | Description        |
| -------------- | ------------------ |
| 0              | Static road edge   |
| 1              | Junction → Vehicle |
| 2              | Vehicle → Junction |
| 3              | Vehicle → Vehicle  |


Other Features
current_edge - of shape: [num_vehicle_nodes], LongTensor of edge indices
position_on_edge - of shape: [num_vehicle_nodes], LongTensor of edge indices
vehicles_route - list of edge indices (route) for each vehicle in [num_vehicle_nodes]

Snapshot pytorch geometric data structure:

| Tensor             | Shape                         | Description                                                |
| ------------------ | ----------------------------- | ---------------------------------------------------------- |
| `x`                | `[N_nodes, F_node]`           | Node features (junctions + vehicles)                       |
| `edge_index`       | `[2, N_edges]`                | Source–target node indices (includes dynamic edges)        |
| `edge_attr`        | `[N_edges, F_edge]`           | Edge feature vectors (e.g., speed, lanes, length, etc.)    |
| `edge_type`        | `[N_edges]`                   | Edge type: int class                                       |
| `current_edge`     | `[N_vehicles]`                | Index of current edge each vehicle is on                   |
| `position_on_edge` | `[N_vehicles]`                | Normalized position (0 to 1) on current edge               |
| `vehicle_ids`      | `list[str]`                   | Original vehicle IDs (for traceability)                    |
| `junction_ids`     | `list[str]`                   | List of junction IDs                                       |
| `edge_ids`         | `list[str]`                   | List of edge IDs                                           |
| `vehicle_routes`   | `list[list[int]]`             | List of edge index sequences (routes) per vehicle          |
| ------------------ | ----------------------------- | ---------------------------------------------------------- |
| `y`                | `[N_vehicles]`                | Target label (ETA) per vehicle                             |
| ------------------ | ----------------------------- | ---------------------------------------------------------- |



'''
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


class DatasetCreator:
    def __init__(
            self,
            config,
            snapshots_folder,
            labels_folder,
            eda_folder,
            out_graph_folder,
            filter_outliers,
            z_normalize,
            log_normalize,
            mapping_folder
        ):
        # Load configuration
        
        self.config = config
        self.snapshots_folder = snapshots_folder
        self.eda_dir = eda_folder
        self.mapping_dir = mapping_folder
        self.labels_folder = labels_folder
        self.snapshot_files = sorted([f for f in os.listdir(self.snapshots_folder) if f.endswith('.json') and "labels" not in f])
        self.out_graph_folder = out_graph_folder
        if not os.path.exists(self.out_graph_folder):
            os.makedirs(self.out_graph_folder)
        self.vehicle_mapping_file = os.path.join(self.mapping_dir, "vehicle_mapping.json")
        self.junction_mapping_file = os.path.join(self.mapping_dir, "junction_mapping.json")
        self.edge_mapping_file = os.path.join(self.mapping_dir, "edge_mapping.json")

        self.edge_features_file = os.path.join(self.eda_dir, "edge_feature_summary.csv")
        self.vehicle_features_file = os.path.join(self.eda_dir, "vehicle_feature_summary.csv")
        self.junction_features_file = os.path.join(self.eda_dir, "junction_feature_summary.csv")
        self.label_features_file = os.path.join(self.eda_dir, "labels_feature_summary.csv")
        self.edge_route_count_file = os.path.join(self.eda_dir, "edge_route_count_summary.csv")
        
        if not self.check_required_files():
            raise FileNotFoundError("Required files for dataset creation are missing. Please ensure all necessary files are present in the specified directories.")
        with open(config) as f:
            self.config = json.load(f)
        self.filter_outliers = filter_outliers
        self.z_normalize = z_normalize
        self.log_normalize = log_normalize
        self.entities_data = {}
        self.populate_entities_data()
        self.is_init = True
        # self.print_entities_data()

    def populate_entities_data(self):
        '''
        populate the data with statistical info for normalization and index mapping
        for junction, vehicle, label and edge include:
        1. ids from mapping json
        2. features statistics from summary csv.
            if feature is numeric add its statistical analysis
            if it is categorical add its keys only
        3. all static features like dimensions, number of lanes (for edges) etc...
        '''
        def parse_counts(val):
            try:
                if pd.isna(val) or val == '':
                    return {}
                return ast.literal_eval(val)
            except Exception:
                return {}
            
        entities = ['junction', 'vehicle', 'edge', 'label']
        mappings = [self.junction_mapping_file, 
                    self.vehicle_mapping_file, 
                    self.edge_mapping_file, 
                    None]
        statistics = [self.junction_features_file, 
                      self.vehicle_features_file, 
                      self.edge_features_file,
                      self.label_features_file]
        
        for i, entity in enumerate(entities):
            self.entities_data[entity] = {'ids':[], 'features':{}, 'stats':{}}
            if mappings[i] is not None:
                with open(mappings[i], 'r') as f:
                    data_map = json.load(f)
                    print(f"Loaded {entity} mapping: {len(data_map)} entries")
                    if len(data_map) == 0:
                        print(f"Warning: {entity} mapping file '{mappings[i]}' contains no entries!.")
                        exit(1)

                self.entities_data[entity]['ids'] = sorted(data_map.keys())
                print(f"entity {entity} has {len(self.entities_data[entity]['ids'])} ids")
                print(self.entities_data[entity]['ids'][:5])
                self.entities_data[entity]['features'] = data_map

            df = pd.read_csv(statistics[i])
            for _, row in df.iterrows():
                feature_name = row['feature']
                entry = {}
                # Handle numeric features
                if row['type'] == 'numeric':
                    entry['mean'] = float(row.get('mean', 0.0))
                    entry['std'] = float(row.get('std', 1.0))
                    entry['min'] = float(row.get('min', 0.0))
                    entry['max'] = float(row.get('max', 1.0))

                # Handle categorical features
                elif row['type'] == 'categorical':
                    value_counts = parse_counts(row.get('value_counts', ''))
                    count = row.get('count', 0)
                    if count == 0:
                        print(f"Warning: categorical count = 0 in '{self.junction_features_file}'")
                        exit(1)
                    entry['keys'] = sorted(value_counts.keys()) if value_counts else []

                self.entities_data[entity]['stats'][feature_name] = entry

    def print_entities_data(self):
        print(self.entities_data)

    def check_required_files(self):
        required_files = [
            self.vehicle_mapping_file,
            self.junction_mapping_file,
            self.edge_mapping_file
        ]
        for file in required_files:
            if not os.path.exists(file):
                print(f"Missing required file: {file}")
                return False
        if not os.path.exists(self.snapshots_folder):
            print(f"Missing snapshot directory: {self.snapshots_folder}")
            return False
        if not os.path.exists(self.mapping_dir):
            print(f"Missing mapping directory: {self.mapping_dir}")
            return False
        if not self.snapshot_files or len(self.snapshot_files) == 0:
            print(f"No snapshot files found in directory: {self.snapshots_folder}")
            return False
        if not os.path.exists(self.labels_folder) or not os.listdir(self.labels_folder):
            print(f"No label files found in directory: {self.labels_folder}")
            return False
        if not os.path.exists(self.edge_features_file):
            print(f"Missing edge features file: {self.edge_features_file}")
            return False
        if not os.path.exists(self.vehicle_features_file):
            print(f"Missing vehicle features file: {self.vehicle_features_file}")
            return False
        if not os.path.exists(self.junction_features_file):
            print(f"Missing junction features file: {self.junction_features_file}")
            return False
        if not os.path.exists(self.label_features_file):
            print(f"Missing label features file: {self.label_features_file}")
            return False
        if not os.path.exists(self.edge_route_count_file):
            print(f"Missing edge route count file: {self.edge_route_count_file}")
            return False
        print("All required files and directories are present.")
        return True

    def create_dataset(self):
        print(f"Converting {len(self.snapshot_files)} snapshots to PyG Data objects...")
        for snap_file in tqdm(self.snapshot_files, desc="Processing snapshots"):
            snapshot_path = os.path.join(self.snapshots_folder, snap_file)
            label_file = snap_file.replace("step_", "labels_")
            label_path = os.path.join(self.labels_folder, label_file)
            out_graph_path = os.path.join(self.out_graph_folder, snap_file.replace(".json", ".pt"))



def main():
    parser = argparse.ArgumentParser(description="Create a traffic dataset from simulation snapshots.")
    parser.add_argument(
        '--config', 
        default="simulation.config.json", 
        help="Path to simulation config JSON file."
    )
    
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
        "--z_normalize",
        action="store_true",
        default=True,
        help="Enable z-score normalization of feature and labels (default: True)"
    )

    parser.add_argument(
        "--log_normalize",
        action="store_true",
        default=True,
        help="Enable log + z-score normalization of features (default: True)"
    )

    parser.add_argument(
        "--mapping_folder",
        action="store_true",
        default="eda_exports/mappings",
        help="Folder with mappings for vehicle, junction, and edge features (default: eda_exports/mappings)"
    )

    args = parser.parse_args()

    os.makedirs(args.out_graph_folder, exist_ok=True)

    # Create dataset creator instance
    creator = DatasetCreator(
        args.config, 
        args.snapshots_folder,
        args.labels_folder,
        args.eda_folder,
        args.out_graph_folder,
        args.filter_outliers,
        args.z_normalize,
        args.log_normalize,
        args.mapping_folder
        )
    creator.create_dataset()


if __name__ == "__main__":
    main()