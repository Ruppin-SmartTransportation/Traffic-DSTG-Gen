"""
x - Nodes (Junctions, Vehicle) main feature layout:

| Index | Feature Name        | Notes                                                   |
| ----- | ------------------- | ------------------------------------------------------- |
| 0     | `node_type`         | 0 = junction    1 = vehicle                             |
| 1-3   | `veh_type_oh`       | ['bus', 'passenger', 'truck']`[0, 0, 0]` for junctions  |
| 4     | `speed`             | min-max normalized if normalize, else raw               |
| 5     | `acceleration`      | min-max normalized if normalize, else raw               |
| 6     | `sin_hour`          | represent time in a unit circle                         |  
| 7     | `cos_hour`          | represent time in a unit circle                         |  
| 8     | `sin_day`           | represent day in a unit circle                          |  
| 9     | `cos_day`           | represent day in a unit circle                          |  
| 10    | `route_length`      | min-max normalized if normalize, else raw               |
| 11    | `progress`          | trip progress: 1 - (route_length_left / route_length)   |
| 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
| 16    | `current_x`         | min-max normalized if normalize, else raw               |
| 17    | `current_y`         | min-max normalized if normalize, else raw               |
| 18    | `destination_x`     | Normalized or raw; for vehicles only                     |
| 19    | `destination_y`     | Normalized or raw; for vehicles only                     |
| 20-22 | `current_edge_num_lanes_oh` | One-hot: [1,2,3] lanes; [0,0,0] for junctions         |
| 23    | `current_edge_demand`     | Demand value for the current edge (from updated edge features) |
| 24    | `current_edge_occupancy`  | Occupancy value for the current edge (from updated edge features) |
| 25    | `j_type`            | Junction type (priority/traffic_light); 0 for vehicles  |

Total of 26 features per node.

edge_features - Roads (static Edges) main feature layout:

| Index | Feature Name        | Notes                                                   |
| ----- | ---------------------- | ---------------------------------------------------- |
| 0     | `avg_speed`            | min-max normalized if normalize, else raw            |
| 1-3   | `num_lanes`            | One-hot of 1-3 lanes                                 |
| 4     | `length`               | min-max normalized if normalize, else raw            |
| 5     | `edge_demand`          | log+z-score normalized if log_normalize, else min-max if normalize, else raw |
| 6     | `edge_occupancy`       | log+z-score normalized if log_normalize, else min-max if normalize, else raw |

Total of 7 features per edge.

Normalization options:
- If `normalize` is True, min-max normalization is applied to most continuous features: (value - min) / (max - min).
- If `log_normalize` is True, log+z-score normalization is applied to skewed/count features (edge_demand, edge_occupancy): log1p(value) - mean / std.
- If both are False, raw values are used.
- For edge_demand and edge_occupancy, log_normalize takes precedence over normalize.

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
| `edge_attr`        | `[N_edges, F_edge]`           | Edge feature vectors (static and dynamic, see code)        |
| `edge_type`        | `[N_edges]`                   | Edge type: int class                                       |
| `current_edge`     | `[N_vehicles]`                | Index of current edge each vehicle is on                   |
| `position_on_edge` | `[N_vehicles]`                | Normalized position (0 to 1) on current edge               |
| `vehicle_ids`      | `list[str]`                   | Original vehicle IDs (for traceability)                    |
| `junction_ids`     | `list[str]`                   | List of junction IDs                                       |
| `edge_ids`         | `list[str]`                   | List of edge IDs                                           |
| `vehicle_routes`   | `list[list[int]]`             | List of edge index sequences (routes) per vehicle          |
| `y`                | `[N_vehicles]`                | Target label (ETA) per vehicle                             |
| `y_equal_thirds`   | `[N_vehicles]`                | ETA category by equal thirds                               |
| `y_quartile`       | `[N_vehicles]`                | ETA category by quartile                                   |
| `y_mean_pm_0_5_std`| `[N_vehicles]`                | ETA category by mean±0.5std                                |
| `y_median_pm_0_5_iqr`| `[N_vehicles]`              | ETA category by median±0.5IQR                              |
| `y_binary_eta`     | `[N_vehicles]`                | Binary ETA label (short/long)                              |

"""
import os
import json
import torch
import pandas as pd
import numpy as np
import ast
import argparse
from torch_geometric.data import Data
from tqdm import tqdm
from collections import defaultdict
import math
import re
import random

def extract_step_number(filename):
    match = re.search(r"step_(\d+)\.json", filename)
    return int(match.group(1)) if match else -1

def extract_numeric_suffix(s):
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else float('inf')

# Global variable for node feature names (order matters, j_type is last)
NODE_FEATURE_NAMES = [
    'node_type', # 0
    'veh_type_oh_0', 'veh_type_oh_1', 'veh_type_oh_2', # 1-3
    'speed', # 4
    'acceleration', # 5
    'sin_hour', # 6
    'cos_hour', # 7
    'sin_day', # 8
    'cos_day', # 9
    'route_length', # 10
    'progress', # 11
    'zone_oh_0', 'zone_oh_1', 'zone_oh_2', 'zone_oh_3', # 12-15
    'current_x', # 16
    'current_y', # 17
    'destination_x', # 18
    'destination_y', # 19
    'current_edge_num_lanes_oh_0', 'current_edge_num_lanes_oh_1', 'current_edge_num_lanes_oh_2', # 20-22
    'current_edge_demand', # 23
    'current_edge_occupancy',
    'j_type'  # Always last
]

NODE_FEATURES_COUNT = len(NODE_FEATURE_NAMES)
EDGE_FEATURES_COUNT = 7
# Global variable for edge feature names (order matters)
EDGE_FEATURE_NAMES = [
    'avg_speed',
    'num_lanes_oh_0', 'num_lanes_oh_1', 'num_lanes_oh_2',
    'length',
    'edge_demand',
    'edge_occupancy'
]

class DatasetCreator:
    def __init__(
            self,
            config,
            snapshots_folder,
            labels_folder,
            eda_folder,
            out_graph_folder,
            filter_outliers,
            normalize,
            log_normalize,
            mapping_folder,
            eta_analysis_methods_path,
            skip_validation=False,
            fast_validation=False
        ):
        # Load configuration
        
        self.config = config
        self.snapshots_folder = snapshots_folder
        self.eda_dir = eda_folder
        self.mapping_dir = mapping_folder
        self.labels_folder = labels_folder
        self.snapshot_files = sorted(
            [f for f in os.listdir(self.snapshots_folder) if f.startswith("step_") and f.endswith(".json")],
            key=extract_step_number
        )        
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
        self.eta_analysis_methods_path = eta_analysis_methods_path
        
        if not self.check_required_files():
            raise FileNotFoundError("Required files for dataset creation are missing. Please ensure all necessary files are present in the specified directories.")
        with open(config) as f:
            self.config = json.load(f)
        self.filter_outliers = filter_outliers
        self.normalize = normalize
        self.log_normalize = log_normalize
        self.skip_validation = skip_validation
        self.fast_validation = fast_validation
        self.entities_data = {}
        self.populate_entities_data()
        self.is_init = True
        # self.print_entities_data()

        # Load 99th percentile for total_travel_time_seconds from labels_feature_summary.csv
        label_summary_path = os.path.join(self.eda_dir, "labels_feature_summary.csv")
        label_summary_df = pd.read_csv(label_summary_path)
        ttt_row = label_summary_df[label_summary_df['feature'] == 'total_travel_time_seconds']
        if ttt_row.empty or '99%' not in ttt_row:
            raise ValueError("Could not find 99th percentile for total_travel_time_seconds in labels_feature_summary.csv")
        self.travel_time_99p = float(ttt_row['99%'].values[0])
        self.travel_time_min = 180.0  # 3 minutes in seconds

        # Load thresholds for all four methods from eta_analysis_methods.csv
        if not os.path.exists(self.eta_analysis_methods_path):
            raise FileNotFoundError(f"Missing eta_analysis_methods.csv at {self.eta_analysis_methods_path}")
        methods_df = pd.read_csv(self.eta_analysis_methods_path)
        self.method_thresholds = {}
        for _, row in methods_df.iterrows():
            method = row['method']
            self.method_thresholds[method] = {
                'short': float(row['short_threshold_seconds']),
                'long': float(row['long_threshold_seconds'])
            }
        # Map verbose method names to short keys
        self.method_key_map = {
            'Equal Thirds (33.33%)': 'equal_thirds',
            'Quartile-based (25-75)': 'quartile',
            'Mean plus minus 0.5 Std': 'mean_pm_0_5_std',
            'Median plus minus 0.5 IQR': 'median_pm_0_5_iqr',
        }

        # Load median ETA for binary threshold
        label_summary_path = os.path.join(self.eda_dir, "labels_feature_summary.csv")
        label_summary_df = pd.read_csv(label_summary_path)
        eta_row = label_summary_df[label_summary_df['feature'] == 'eta']
        if eta_row.empty or 'median' not in eta_row:
            raise ValueError("Could not find median for eta in labels_feature_summary.csv")
        self.eta_binary_threshold = float(eta_row['median'].values[0])

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
            
        entities = ['junction', 'vehicle', 'edge', 'label', 'edge']
        mappings = [self.junction_mapping_file, 
                    self.vehicle_mapping_file, 
                    self.edge_mapping_file, 
                    None,
                    None]
        statistics = [self.junction_features_file, 
                      self.vehicle_features_file, 
                      self.edge_features_file,
                      self.label_features_file,
                      self.edge_route_count_file]
        
        for i, entity in enumerate(entities):
            if entity not in self.entities_data:
                self.entities_data[entity] = {'ids':[], 'features':{}, 'stats':{}}
            if mappings[i] is not None:
                with open(mappings[i], 'r') as f:
                    data_map = json.load(f)
                    print(f"Loaded {entity} mapping: {len(data_map)} entries")
                    if len(data_map) == 0:
                        print(f"Warning: {entity} mapping file '{mappings[i]}' contains no entries!.")
                        exit(1)
                # Enforce natural sorting for all entities
                self.entities_data[entity]['ids'] = sorted(
                    data_map.keys(),
                    key=extract_numeric_suffix
                )
                print(f"entity {entity} has {len(self.entities_data[entity]['ids'])} ids")
                print(f"self.entities_data['{entity}']['ids'][:20]", self.entities_data[entity]['ids'][:20])
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
                    # add 97 98 99 percentile if available
                    for p in [97, 98, 99]:
                        percentile_name = f'{p}%'
                        if percentile_name in row:
                            entry[percentile_name] = float(row[percentile_name])

                # Handle categorical features
                elif row['type'] == 'categorical':
                    value_counts = parse_counts(row.get('value_counts', ''))
                    count = row.get('count', 0)
                    if count == 0:
                        print(f"Warning: categorical count = 0 in '{statistics[i]}'")
                        exit(1)
                    entry['keys'] = sorted(value_counts.keys()) if value_counts else []

                self.entities_data[entity]['stats'][feature_name] = entry
                print(f"[{entity}]['stats'][{feature_name}]", self.entities_data[entity]['stats'][feature_name])

    def print_entities_data(self):
        print(self.entities_data)

    def check_required_files(self):
        required_files = [
            self.vehicle_mapping_file,
            self.junction_mapping_file,
            self.edge_mapping_file,
            self.label_features_file,
            self.eta_analysis_methods_path
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

    def process_junctions(self):
        """
        Processes junction nodes and constructs their feature vectors.
        The feature vector layout matches NODE_FEATURE_NAMES (length 26), with j_type as the last feature.
        For junctions, vehicle-specific features (e.g., speed, acceleration, destination_x/y, current_edge_num_lanes_oh, current_edge_demand, current_edge_occupancy) are set to 0 or [0,0,0] as appropriate.
        """
        stats = self.entities_data['junction']['stats']
        junction_ids = self.entities_data['junction']['ids']
        if len(junction_ids) == 0:
            print(f"ERROR: No junctions found in {self.junction_mapping_file}. Terminating.")
            exit(1)
        junction_ids_to_index = {jid: idx for idx, jid in enumerate(junction_ids)}
        print(f"Processing {len(junction_ids)} junctions with IDs: {junction_ids[:5]}...")
        static_features = self.entities_data['junction']['features']
        junction_features = []
        for jid in junction_ids:    
            features = [0] * NODE_FEATURES_COUNT  # Initialize with zeros for all features
            # Fill in junction-specific features
            features[0] = 0  # node_type = 0 for junction
            # zone_oh (one-hot encoding)
            zone_oh = [0] * len(stats['zone']['keys'])
            zone = static_features[jid].get('zone', '_')
            if zone not in stats['zone']['keys']:
                print(f"ERROR: Junction {jid} has unknown zone '{zone}'. Terminating.")
                exit(1)
            zone_index = stats['zone']['keys'].index(zone)
            features[12 + zone_index] = 1  # zone_oh at indices 12-15
            # current_x, current_y
            x = static_features[jid].get('x', 0.0)
            y = static_features[jid].get('y', 0.0)
            if x is None or y is None:
                print(f"ERROR: Junction {jid} has no x or y defined. Terminating.")
                exit(1)
            if self.normalize:
                x = (x - stats['x']['min']) / (stats['x']['max'] - stats['x']['min'])
                y = (y - stats['y']['min']) / (stats['y']['max'] - stats['y']['min'])
            features[16] = x
            features[17] = y
            # j_type (junction type) as last feature
            j_type = static_features[jid].get('type', 'unknown')
            if j_type not in stats['type']['keys']:
                print(f"ERROR: Junction {jid} has unknown type '{j_type}'. Terminating.")
                exit(1)
            features[NODE_FEATURES_COUNT - 1] = stats['type']['keys'].index(j_type)
            # All other features (vehicle-specific) remain 0
            junction_features.append(features)
        print(f"Processed {len(junction_features)} junction features and create {len(junction_ids_to_index.keys())} indices.")

        return junction_ids_to_index, junction_features
    
    def process_static_edges(self, junction_ids_to_index):
        """
        Processes edge data into a map and extracts features from the edge feature summary CSV.

        Returns:
            static_edge_index (list): List of edge indices in the format [[src_node_0, src_node_1, ...], [tgt_node_0, tgt_node_1, ...]]
            static_edge_type (list): List of edge types corresponding to the edges in edge_index
            static_edge_ids_to_index (dict): {edge_id: index}
            static_edge_features (list): List of edge feature vectors

        edge_features - Roads (static Edges) main feature layout:

        | Index | Feature Name        | Notes                                                   |

        | ----- | ---------------------- | ---------------------------------------------------- |
        | 0     | `avg_speed`            | normalized                                         |
        | 1-3   | `num_lanes`            | One-hot of 1-3 lanes                                 |
        | 4     | `length`               | normalized (distance between from and to nodes)    |
        | 5     | `edge_demand`          | log + z-score normalized                             |
        | 6     | `edge_occupancy`       | normalized                                         |

        Total of 7 features per edge.

        edges_ids_to_idx = {'AX3AX2': 0, 'AX4AX3': 1, ...} (only static edges)
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

        """
        stats = self.entities_data['edge']['stats']
        print(f"Found {len(stats)} Stats for edges")
        print(f"Edge stats keys: {list(stats.keys())}")
        static_features = self.entities_data['edge']['features']
        print(f"Found {len(static_features)} Static Features for edges")
        edge_ids = self.entities_data['edge']['ids']
        print(f"Found {len(edge_ids)} Edge IDs: {edge_ids[:5]}...")
        if len(edge_ids) == 0:
            print(f"ERROR: No edges found in {self.edge_mapping_file}. Terminating.")
            exit(1)
        # construct static edge index
        static_edge_ids_to_index = {eid: idx for idx, eid in enumerate(edge_ids)}
        static_edge_type = [0] * len(edge_ids)  # All static edges are type 0
        static_edge_index = [[], []]
        for eid in edge_ids:
            # Get the source and target junction IDs from the static features
            src_junction = static_features[eid].get('from', None)
            tgt_junction = static_features[eid].get('to', None)
            if src_junction is None or tgt_junction is None:
                print(f"ERROR: Edge {eid} has no 'from' or 'to' defined. Terminating.")
                exit(1)
            if src_junction not in self.entities_data['junction']['ids'] or tgt_junction not in self.entities_data['junction']['ids']:
                print(f"ERROR: Edge {eid} has unknown junctions '{src_junction}' or '{tgt_junction}'. Terminating.")
                exit(1)
            # Get the indices of the source and target junctions from static_edge_ids_to_index
            src_index = junction_ids_to_index[src_junction]
            tgt_index = junction_ids_to_index[tgt_junction]
            static_edge_index[0].append(src_index)
            static_edge_index[1].append(tgt_index)
        
        static_features = self.entities_data['edge']['features']
        static_edge_features = []
        for eid in edge_ids:
            features = [] # Initialize with empty list
            # edge feature are avg_speed, num_lanes (one hot), edge_demand, edge_occupancy, length
            avg_speed = 0.0
            features.append(avg_speed)  # Placeholder for avg_speed

            num_lanes = static_features[eid].get('num_lanes', None)
            if num_lanes is None:
                print(f"ERROR: Edge {eid} has no num_lanes defined. Terminating.")
                exit(1)
            if num_lanes not in stats['num_lanes']['keys']:
                print(f"ERROR: Edge {eid} has unknown num_lanes '{num_lanes}' where stats keys are {stats['num_lanes']['keys']}. Terminating.")
                exit(1)
            num_lanes_index = stats['num_lanes']['keys'].index(num_lanes)
            num_lanes_oh = [0] * len(stats['num_lanes']['keys'])  # One-hot encoding for num_lanes
            num_lanes_oh[num_lanes_index] = 1  # Set the correct index to 1
            features.extend(num_lanes_oh)  # Add one-hot num_lanes features
            length = static_features[eid].get('length', -1.0)
            if length == -1.0:
                print(f"ERROR: Edge {eid} has no length defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                length = length / stats['length']['max']
            features.append(length)
            # edge_demand and edge_occupancy will be calculated dynamically per snapshot
            features.append(0)  # Placeholder for edge_demand
            features.append(0)  # Placeholder for edge_occupancy
            assert len(features) == 7, f"Edge features for {eid} should have 7 elements, got {len(features)}"
            static_edge_features.append(features)
        
        return  static_edge_index, static_edge_type, static_edge_ids_to_index, static_edge_features

    def get_eta_category(self, eta, short_th, long_th):
        if eta < short_th:
            return 0  # short
        elif eta < long_th:
            return 1  # medium
        else:
            return 2  # long

    def process_labels(self, snap_file):
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
        label_file = snap_file.replace("step_", "labels_")
        label_path = os.path.join(self.labels_folder, label_file)
        with open(label_path, 'r') as f:
                label_data = json.load(f)

        label_stats = self.entities_data['label']['stats']
        eta_stats = label_stats.get("eta", {})
        if not eta_stats:
            print(f"Warning: No 'eta' stats found in {self.label_features_file}. Terminating.")
            exit(1)
        
        filtered_label_map = {}
        filtered_vehicle_ids = set()
        y_cats = {k: [] for k in self.method_key_map.values()}
        binary_threshold = self.eta_binary_threshold
        y_binary = []
        for entry in label_data:
            eta = entry.get('eta')
            vid = entry.get('vehicle_id')
            duration = entry.get('total_travel_time_seconds', 0)
            if eta is None or vid is None:
                print(f"Warning: Missing 'eta' or 'vehicle_id' in label entry {entry}. Terminating.")
                exit(1)
            if not (self.travel_time_min <= duration <= self.travel_time_99p):
                continue  # skip out-of-range
            if self.normalize :
                # Use filtered range for normalization: 180 to 99th percentile (4732)
                eta_min_filtered = self.travel_time_min
                eta_max_filtered = self.travel_time_99p  # 4732.0
                eta_final = (eta - eta_min_filtered) / (eta_max_filtered - eta_min_filtered)
            else:
                eta_final = eta
            filtered_label_map[vid] = eta_final
            filtered_vehicle_ids.add(vid)
            # Compute all label categories for this vehicle
            for verbose, short_key in self.method_key_map.items():
                th = self.method_thresholds[verbose]
                y_cats[short_key].append(self.get_eta_category(eta, th['short'], th['long']))
            # Binary label: 0 if short, 1 if long
            y_binary.append(0 if eta < binary_threshold else 1)
        if len(filtered_label_map) != len(filtered_vehicle_ids):
            print(f"Warning: Mismatch in filtered_label_map and filtered_vehicle_ids length. {len(filtered_label_map)} vs {len(filtered_vehicle_ids)}")
            exit(1)
        filtered_vehicle_ids = sorted(filtered_vehicle_ids, key=extract_numeric_suffix)
        etas = [filtered_label_map[vid] for vid in filtered_vehicle_ids]
        y = torch.FloatTensor(etas)
        # Convert y_cats to tensors and validate
        y_cat_tensors = {}
        for k, v in y_cats.items():
            t = torch.LongTensor([v[i] for i, vid in enumerate(filtered_vehicle_ids)]) if len(v) == len(filtered_vehicle_ids) else torch.LongTensor(v)
            if len(t) != len(filtered_vehicle_ids):
                raise ValueError(f"Mismatch: y_{k} ({len(t)}) vs filtered_vehicle_ids ({len(filtered_vehicle_ids)})")
            y_cat_tensors[k] = t
        y_binary_tensor = torch.LongTensor(y_binary)
        if len(y_binary_tensor) != len(filtered_vehicle_ids):
            raise ValueError(f"Mismatch: y_binary ({len(y_binary_tensor)}) vs filtered_vehicle_ids ({len(filtered_vehicle_ids)})")
        return y, filtered_vehicle_ids, y_cat_tensors, y_binary_tensor

    def process_vehicle_features(self, snap_file, current_vehicle_ids, static_edge_ids_to_index, static_edge_features):
        """
        Processes vehicle features from a snapshot file and returns a list of feature vectors.

        Args:
            snap_file (str): Path to the snapshot JSON file.
            current_vehicle_ids (list): List of vehicle IDs present in the snapshot.

        Returns:
            current_vehicle_features (list): List of feature vectors for each vehicle.
            snapshot_data (dict): Parsed snapshot data.

        entity vehicle has 241269 ids
        self.entities_data['vehicle']['ids'][:5] ['veh_0', 'veh_1', 'veh_2', 'veh_3', 'veh_4']
        [vehicle]['stats'][id] {'keys': []}
        [vehicle]['stats'][node_type] {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0}
        [vehicle]['stats'][vehicle_type] {'keys': ['bus', 'passenger', 'truck']}
        [vehicle]['stats'][length] {'mean': 6.21186330417451, 'std': 2.8288303833139823, 'min': 4.5, 'max': 12.0}
        [vehicle]['stats'][width] {'mean': 2.010606521979847, 'std': 0.3210444355783385, 'min': 1.8, 'max': 2.5}
        [vehicle]['stats'][height] {'mean': 1.9842412246705772, 'std': 0.7401846131272568, 'min': 1.5, 'max': 3.2}
        [vehicle]['stats'][speed] {'mean': 8.14055705469758, 'std': 10.013811309079047, 'min': 0.0, 'max': 33.329011540575884}
        [vehicle]['stats'][acceleration] {'mean': -0.0594831348611925, 'std': 0.8681193250319945, 'min': -4.561605298881993, 'max': 2.6}
        [vehicle]['stats'][current_x] {'mean': 8362.154813936202, 'std': 4173.723850538964, 'min': -4.8, 'max': 18004.8}
        [vehicle]['stats'][current_y] {'mean': -1094.4905685203237, 'std': 2623.0293288565254, 'min': -6269.76, 'max': 5004.8}
        [vehicle]['stats'][current_zone] {'keys': ['A', 'B', 'C', 'H']}
        [vehicle]['stats'][current_edge] {'keys': []}
        [vehicle]['stats'][current_position] {'mean': 1551.2740315539766, 'std': 2076.182099547493, 'min': 0.0010035338711276, 'max': 6331.846837013244}
        [vehicle]['stats'][origin_name] {'keys': ['home', 'restaurantA', 'restaurantB', 'restaurantC', 'work']}
        [vehicle]['stats'][origin_zone] {'keys': ['A', 'B', 'C']}
        [vehicle]['stats'][origin_edge] {'keys': []}
        [vehicle]['stats'][origin_position] {'mean': 243.9827577864734, 'std': 139.92457280574254, 'min': 1.0020456355236576, 'max': 484.5999408235933}
        [vehicle]['stats'][origin_x] {'mean': 8760.170384683235, 'std': 4394.625914910833, 'min': 11.429063726717914, 'max': 17988.34598349441}
        [vehicle]['stats'][origin_y] {'mean': -1362.001583582693, 'std': 3528.7149174482247, 'min': -6253.50715848932, 'max': 4988.506738605275}
        [vehicle]['stats'][origin_start_sec] {'mean': 1159695.0522159785, 'std': 689817.331645224, 'min': 1.0, 'max': 2419021.0}
        [vehicle]['stats'][route_length] {'mean': 12872.689068334075, 'std': 3540.920668086482, 'min': 482.4, 'max': 23128.59}
        [vehicle]['stats'][route_length_left] {'mean': 8140.859801807668, 'std': 4212.555102347426, 'min': 482.4, 'max': 22642.99}
        [vehicle]['stats'][destination_name] {'keys': ['friend1', 'friend2', 'friend3', 'home', 'park1', 'park2', 'park3', 'park4', 'restaurantA', 'restaurantB', 'restaurantC', 'stadium1', 'stadium2', 'work']}
        [vehicle]['stats'][destination_edge] {'keys': []}
        [vehicle]['stats'][destination_position] {'mean': 241.93812259582415, 'std': 139.00837683866297, 'min': 1.0009821241804862, 'max': 484.599954998163}
        [vehicle]['stats'][destination_x] {'mean': 8713.34363136051, 'std': 5475.499328714149, 'min': 11.549815041727946, 'max': 17987.947945205477}
        [vehicle]['stats'][destination_y] {'mean': -30.57885467859064, 'std': 3429.891997666032, 'min': -6253.498252087569, 'max': 4988.425209122459}

        | Index | Feature Name        | Notes                                                   |
        | ----- | ------------------- | ------------------------------------------------------- |
        | 0     | `node_type`         | 0 = junction    1 = vehicle                             |
        | 1-3   | `veh_type_oh`       | ['bus', 'passenger', 'truck']`[0, 0, 0]` for junctions  |
        | 4     | `speed`             | z normalized                                            |
        | 5     | `acceleration`      | z normalized                                            |
        | 6     | `sin_hour`          | represent time in a unit circle                         |
        | 7     | `cos_hour`          | represent time in a unit circle                         |
        | 8     | `sin_day`           | represent day in a unit circle                          |
        | 9     | `cos_day`           | represent day in a unit circle                          |
        | 10    | `route_length`      | z normalized                                            |
        | 11    | `progress`          | trip progress: 1 - (route_length_left / route_length)   |
        | 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
        | 16    | `current_x`         | z normalized                                            |
        | 17    | `current_y`         | z normalized                                            |
        | 18    | `destination_x`     | normalized destination x coordinate                      |
        | 19    | `destination_y`     | normalized destination y coordinate                      |
        | 20-22 | `current_edge_num_lanes_oh` | one-hot encoding of current edge's num_lanes (1, 2, or 3 lanes); [0,0,0] for junctions |
        | 23    | `current_edge_demand`     | Demand value for the current edge (from updated edge features) |
        | 24    | `current_edge_occupancy`  | Occupancy value for the current edge (from updated edge features) |
        | 25    | `j_type`            | Junction type (priority/traffic_light); 0 for vehicles  |

        Total of 26 features per node.

        """
        snap_path = os.path.join(self.snapshots_folder, snap_file)
        if not os.path.exists(snap_path):
            print(f"ERROR: Snapshot file {snap_path} does not exist. Terminating.")
            exit(1)
        with open(snap_path, 'r') as f:
            snapshot_data = json.load(f)
        vehicle_stats = self.entities_data['vehicle']['stats']
        nodes = snapshot_data.get('nodes', [])
        if not nodes:
            print(f"ERROR: Snapshot {snap_file} has no nodes defined. Terminating.")
            exit(1)
        vehicle_nodes = [node for node in nodes if node.get('node_type') == 1]  # Filter for vehicle nodes
        if not vehicle_nodes:
            print(f"ERROR: Snapshot {snap_file} has no vehicle nodes defined. Terminating.")
            exit(1)
        if len(vehicle_nodes) < len(current_vehicle_ids):
            print(f"ERROR: Snapshot {snap_file} has fewer vehicle nodes ({len(vehicle_nodes)}) than expected ({len(current_vehicle_ids)}). Terminating.")
            exit(1)
        #sort vehicle nodes by their IDs to ensure consistent order
        vehicle_nodes = sorted(vehicle_nodes, key=lambda x: extract_numeric_suffix(x.get('id', '')))
        features = []
        vehicle_routes_flat = []
        vehicle_route_splits = []
        current_edges = []
        position_on_edges = []

        for vid in current_vehicle_ids:
            vehicle_node = next((node for node in vehicle_nodes if node.get('id') == vid), None)
            if vehicle_node is None:
                print(f"ERROR: Vehicle {vid} not found in snapshot {snap_file}. Terminating.")
                exit(1)
            # 1. Feature extraction for each vehicle node
            feature = []  # Initialize feature vector
            # node_type = 1 for vehicle
            if 'node_type' not in vehicle_node:
                print(f"ERROR: Vehicle {vid} node does not have 'node_type' defined. Terminating.")
                exit(1)
            if vehicle_node['node_type'] != 1:
                print(f"ERROR: Vehicle {vid} node_type is not 1 (vehicle). Found: {vehicle_node['node_type']}. Terminating.")
                exit(1)
            feature.append(1)  # [0] node_type = 1 for vehicle
            if 'vehicle_type' not in vehicle_node:
                print(f"ERROR: Vehicle {vid} node does not have 'vehicle_type' defined. Terminating.")
                exit(1)
            if vehicle_node['vehicle_type'] not in vehicle_stats['vehicle_type']['keys']:
                print(f"ERROR: Vehicle {vid} has unknown vehicle_type '{vehicle_node['vehicle_type']}'. Terminating.")
                exit(1)
            # vehicle_type is one-hot encoded
            # veh_type_oh ['bus', 'passenger', 'truck']
            veh_type_oh = [0] * len(vehicle_stats['vehicle_type']['keys'])  # One-hot encoding for vehicle type
            veh_type = vehicle_node['vehicle_type']
            if veh_type is None:
                print(f"ERROR: Vehicle {vid} has no vehicle_type defined. Terminating.")
                exit(1)
            if veh_type not in vehicle_stats['vehicle_type']['keys']:
                print(f"ERROR: Vehicle {vid} has unknown vehicle_type '{veh_type}'. Terminating.")
                exit(1)
            # Get the index of the vehicle type in the keys
            veh_type_index = vehicle_stats['vehicle_type']['keys'].index(veh_type)
            veh_type_oh[veh_type_index] = 1  # Set the correct vehicle type index to 1
            feature.extend(veh_type_oh)  # [1-3] Add one-hot vehicle type features
            # speed
            speed = vehicle_node.get('speed', None)
            if speed is None:
                print(f"ERROR: Vehicle {vid} has no speed defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                speed = (speed - vehicle_stats['speed']['min']) / (vehicle_stats['speed']['max'] - vehicle_stats['speed']['min'])
            feature.append(speed) # [4]
            # acceleration
            acceleration = vehicle_node.get('acceleration', None)
            if acceleration is None:
                print(f"ERROR: Vehicle {vid} has no acceleration defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                acceleration = (acceleration - vehicle_stats['acceleration']['min']) / (vehicle_stats['acceleration']['max'] - vehicle_stats['acceleration']['min'])   
            feature.append(acceleration) # [5]
            # sin_hour, cos_hour, sin_day, cos_day
            timestamp = snapshot_data.get('step', -1) # step is the timestamp in seconds in steps of 60 seconds
            if timestamp == -1 or not isinstance(timestamp, (int, float)):
                print(f"ERROR: Snapshot {snap_file} has no valid timestamp. Terminating.")
                exit(1)
            minutes = timestamp // 60
            hours = minutes // 60
            days = hours // 24
            day = days % 7  # Get the day of the week (0-6)
            hour = hours % 24  # Get the hour of the day (0-23)
            minutes = minutes % 60  # Get the minutes (0-59)
            hour = (hour + minutes / 60) % 24  # Convert to hour in the range [0, 24)
            feature.append(math.sin(2 * math.pi * hour / 24))  # sin_hour [6]
            feature.append(math.cos(2 * math.pi * hour / 24))  # cos_hour [7]
            feature.append(math.sin(2 * math.pi * day / 7))  # sin_day [8]
            feature.append(math.cos(2 * math.pi * day / 7))  # cos_day [9]
            # route_length 
            route_length = vehicle_node.get('route_length', None)
            if route_length is None:
                print(f"ERROR: Vehicle {vid} has no route_length defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                route_length_norm = (route_length - vehicle_stats['route_length']['min']) / (vehicle_stats['route_length']['max'] - vehicle_stats['route_length']['min'])
            feature.append(route_length_norm) # [10]
            # progress (1 - route_length_left / route_length)
            route_length_left = vehicle_node.get('route_length_left', None)
            if route_length_left is None:
                print(f"ERROR: Vehicle {vid} has no route_length_left defined. Terminating.")
                exit(1)
            if route_length == 0:
                print(f"ERROR: Vehicle {vid} has no route_length defined but has route left. Terminating.")
                exit(1)
            else:
                progress = 1.0 - (route_length_left / route_length)
            feature.append(progress) # [11]
            # zone_oh
            zone_oh = [0] * len(vehicle_stats['current_zone']['keys'])
            current_zone = vehicle_node.get('current_zone', None)
            if current_zone is None:
                print(f"ERROR: Vehicle {vid} has no current_zone defined. Terminating.")
                exit(1)
            if current_zone not in vehicle_stats['current_zone']['keys']:
                print(f"ERROR: Vehicle {vid} has unknown current_zone '{current_zone}'. Terminating.")
                exit(1)
            zone_index = vehicle_stats['current_zone']['keys'].index(current_zone)
            zone_oh[zone_index] = 1  # Set the correct zone index to 1
            feature.extend(zone_oh)  # Add one-hot zone features [12-15]
            # current_x, current_y
            current_x = vehicle_node.get('current_x', None)
            current_y = vehicle_node.get('current_y', None)
            if current_x is None or current_y is None:
                print(f"ERROR: Vehicle {vid} has no current_x or current_y defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                current_x = (current_x - vehicle_stats['current_x']['min']) / (vehicle_stats['current_x']['max'] - vehicle_stats['current_x']['min'])
                current_y = (current_y - vehicle_stats['current_y']['min']) / (vehicle_stats['current_y']['max'] - vehicle_stats['current_y']['min'])
            feature.append(current_x) # [16]
            feature.append(current_y) # [17]
            # j_type is the junction type, not relevant for vehicles, but we need to add it to the feature vector set to 0
            # destination_x
            destination_x = vehicle_node.get('destination_x', None)
            if destination_x is None:
                print(f"ERROR: Vehicle {vid} has no destination_x defined. Terminating.")
                exit(1)
            if self.normalize:
                destination_x = (destination_x - vehicle_stats['destination_x']['min']) / (vehicle_stats['destination_x']['max'] - vehicle_stats['destination_x']['min'])
            feature.append(destination_x) # [18]
            # destination_y
            destination_y = vehicle_node.get('destination_y', None)
            if destination_y is None:
                print(f"ERROR: Vehicle {vid} has no destination_y defined. Terminating.")
                exit(1)
            if self.normalize:
                destination_y = (destination_y - vehicle_stats['destination_y']['min']) / (vehicle_stats['destination_y']['max'] - vehicle_stats['destination_y']['min'])
            feature.append(destination_y) # [19]
            # current edge num_lanes (one-hot)
            current_edge_str = vehicle_node.get('current_edge', None)
            if current_edge_str is None:
                print(f"ERROR: Vehicle {vid} has no current_edge defined. Terminating.")
                exit(1)
            # Get num_lanes for the current edge
            edge_features_map = self.entities_data['edge']['features']
            edge_stats = self.entities_data['edge']['stats']
            if current_edge_str not in edge_features_map:
                print(f"ERROR: Current edge {current_edge_str} not found in edge features map. Terminating.")
                exit(1)
            num_lanes = edge_features_map[current_edge_str].get('num_lanes', None)
            if num_lanes is None:
                print(f"ERROR: Edge {current_edge_str} has no num_lanes defined. Terminating.")
                exit(1)
            num_lanes_keys = edge_stats['num_lanes']['keys']
            if num_lanes not in num_lanes_keys:
                print(f"ERROR: Edge {current_edge_str} has unknown num_lanes '{num_lanes}'. Terminating.")
                exit(1)
            num_lanes_oh = [0] * len(num_lanes_keys)
            num_lanes_index = num_lanes_keys.index(num_lanes)
            num_lanes_oh[num_lanes_index] = 1
            feature.extend(num_lanes_oh)  # [21-22] 
            feature.append(0.0)  # current_edge_demand placeholder [23]
            feature.append(0.0)  # current_edge_occupancy placeholder [24]
            feature.append(0)    # j_type (index 25, always 0 for vehicles)
            # Add the feature vector to the list
            features.append(feature)

            # 2. Extract route and current edge information
            route_edge_ids = vehicle_node.get('route', [])
            route_edge_indices = []
            for eid in route_edge_ids:
                if eid not in static_edge_ids_to_index:
                    print(f"ERROR: Edge ID {eid} not found in static_edge_ids_to_index. Terminating.")
                    exit(1)
                route_edge_indices.append(static_edge_ids_to_index[eid])

            vehicle_routes_flat.extend(route_edge_indices)
            vehicle_route_splits.append(len(route_edge_indices))

            # 3. current edge index
            current_edge_str = vehicle_node.get("current_edge")
            current_edge_idx = static_edge_ids_to_index.get(current_edge_str, -1)
            if current_edge_idx == -1:
                print(f"ERROR: Current Edge ID {current_edge_str} not found in static_edge_ids_to_index. Terminating.")
                exit(1)
            current_edges.append(current_edge_idx)

            # 4. position on edge
            position_on_edge = float(vehicle_node.get("current_position", -1))
            static_edge_features = self.entities_data['edge']['features'][current_edge_str]
            edge_length = static_edge_features.get('length', -1.0)
            if edge_length == -1.0:
                print(f"ERROR: Edge {static_edge_features} has no length defined. Terminating.")
                exit(1)
            if edge_length <= 0:
                print(f"ERROR: Edge {static_edge_features} has non-positive length {edge_length}. Terminating.")
                exit(1)
            if position_on_edge == -1:
                print(f"ERROR: Vehicle {vid} has no current_position defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                # Get the static features for this edge
                position_on_edge = position_on_edge / edge_length  # Normalize position on edge
            if position_on_edge < 0 or position_on_edge > 1:
                print(f"ERROR: Position on edge for vehicle {vid} is out of bounds: {position_on_edge}. Terminating.")
                exit(1)
        
            position_on_edges.append(position_on_edge)
            
        return snapshot_data, features, vehicle_routes_flat, vehicle_route_splits, current_edges, position_on_edges

    def update_edge_features(self, snapshot_data, current_vehicle_ids, static_edge_features):

        """
        Updates edge features based on the current vehicle and dynamic attributes.
        | Index | Feature Name        | Notes                                                   |

        | ----- | ---------------------- | ---------------------------------------------------- |
        | 0     | `avg_speed`            | z normalized                                         |
        | 5     | `edge_demand`          | log + z-score normalized                             |
        | 6     | `edge_occupancy`       | log + z-score normalized                                         |

        Args:
            current_vehicle_ids (list): List of vehicle IDs present in the snapshot.
            static_edge_features (list): List of static edge feature vectors.

        Returns:
            updated_edge_features (list): List of updated edge feature vectors.
        """
        # start with creating an edge demand map for each edge
        edge_demand_map = {eid: 0 for eid in self.entities_data['edge']['ids']}
        # Iterate over the current vehicle IDs and update the edge demand map
        for vid in current_vehicle_ids:
            vehicle_node = next((node for node in snapshot_data.get('nodes', []) if node.get('id') == vid), None)
            if vehicle_node is None:
                print(f"ERROR: Vehicle {vid} not found in snapshot data. Terminating.")
                exit(1)
            # FIXED: Use route_left instead of route for future demand calculation
            # route_left represents the remaining route from current position to destination
            # route represents the complete route from origin to destination
            route_left = vehicle_node.get('route_left', [])
            if not route_left:
                continue
            # add 1 for each edge in the remaining route of the vehicle (this represents future demand)
            for edge_id in route_left:
                if edge_id not in edge_demand_map:
                    print(f"ERROR: Edge {edge_id} not found in edge demand map. Terminating.")
                    exit(1)
                # Increment the demand for this edge
                edge_demand_map[edge_id] += 1
        
        updated_edge_features = []
        # Go through each edge in the snapshot_data and update its features
        edges = snapshot_data.get('edges', [])
        if not edges:
            print(f"ERROR: Snapshot data has no edges defined. Terminating.")
            exit(1)
        edge_ids = self.entities_data['edge']['ids']
        # iterate over edge_ids and make sure all edge_ids are in the edges list
        for edge_id in edge_ids:
            edge = next((e for e in edges if e.get('id') == edge_id), None)
            if edge is None:
                print(f"ERROR: Edge {edge_id} not found in snapshot data. Terminating.")
                exit(1)
            # Get the static features for this edge
            static_features = static_edge_features[self.entities_data['edge']['ids'].index(edge_id)]
            if len(static_features) != 7:
                print(f"ERROR: Static features for edge {edge_id} should have 7 elements, got {len(static_features)}. Terminating.")
                exit(1)
            updated_features = static_features.copy()
            # set the current average speed
            current_avg_speed = edge.get('avg_speed', -1.0)
            if current_avg_speed == -1.0:
                print(f"ERROR: Edge {edge_id} has no avg_speed defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                current_avg_speed = (current_avg_speed - self.entities_data['edge']['stats']['avg_speed']['min']) / (self.entities_data['edge']['stats']['avg_speed']['max'] - self.entities_data['edge']['stats']['avg_speed']['min'])
                
            updated_features[0] = current_avg_speed  # Update avg_speed in the features

            # Calculate edge_demand based on the edge_demand_map (future demand from remaining routes)
            edge_demand = edge_demand_map.get(edge_id, -1)
            if edge_demand <= -1:
                print(f"ERROR: Edge {edge_id} not found in edge demand map. Terminating.")
                exit(1)
            if self.log_normalize:
                edge_demand = math.log1p(edge_demand) - self.entities_data['edge']['stats']['edge_route_count_log']['mean'] / self.entities_data['edge']['stats']['edge_route_count_log']['std']
            elif self.normalize:
                edge_demand = (edge_demand - self.entities_data['edge']['stats']['edge_route_count']['mean']) / self.entities_data['edge']['stats']['edge_route_count']['std']
            else:
                edge_demand = edge_demand
            updated_features[5] = edge_demand  # Update edge_demand in the features
            
            # Calculate edge_occupancy based on number of vehicles currently on the edge
            vehicles_on_road = edge.get('vehicles_on_road', [])
            number_of_vehicles_on_road = len(vehicles_on_road)
            if self.log_normalize:
                edge_occupancy = math.log1p(number_of_vehicles_on_road) -self.entities_data['edge']['stats']['vehicles_on_road_count_log']['mean']/ self.entities_data['edge']['stats']['vehicles_on_road_count_log']['std']
            elif self.normalize:
                edge_occupancy = (number_of_vehicles_on_road - self.entities_data['edge']['stats']['vehicles_on_road_count']['mean']) / self.entities_data['edge']['stats']['vehicles_on_road_count']['std']
            else:
                edge_occupancy = number_of_vehicles_on_road
            updated_features[6] = edge_occupancy  # Update edge_occupancy in the features
            updated_edge_features.append(updated_features)
        return updated_edge_features
    from collections import defaultdict

    def construct_dynamic_edges(
        self,
        current_vehicle_ids,
        current_vehicle_current_edges,
        current_vehicle_position_on_edges,
        static_edge_ids_to_index,
        static_junction_ids_to_index,
        static_edge_features,
        static_edge_info,  # mapping edge_id → {from, to}
        edge_feature_dim,
        junction_offset  # = 0
    ):
        """
        Constructs dynamic edges to represent the relative position of vehicles along their current edge.

        These edges are added on top of the static road graph to model the *current dynamic structure* of traffic flow.

        Returns:
            dynamic_edge_index: [2, N] list of source–target node indices for dynamic edges.
            dynamic_edge_type: [N] list of integer type codes for each edge (1 = J→V, 2 = V→J, 3 = V→V).
            dynamic_edge_attr: [N, F] dummy feature vectors for each dynamic edge.
        """

        # Initialize empty lists for edge_index, edge_type, and edge_attr
        dynamic_edge_index = [[], []]
        dynamic_edge_type = []
        dynamic_edge_attr = []

        # Create a mapping: edge_id → list of vehicle indices currently on that edge
        edge_to_vehicle_map = defaultdict(list)
        for i, edge_idx in enumerate(current_vehicle_current_edges):
            # Get the string edge_id from index
            edge_id = list(static_edge_ids_to_index.keys())[edge_idx.item()]
            edge_to_vehicle_map[edge_id].append(i)  # i = vehicle index (in this batch)

        # Iterate over each road segment (edge) with active vehicles
        for edge_id, veh_indices in edge_to_vehicle_map.items():
            # Get basic info for this edge
            edge_static = static_edge_info[edge_id]  # dict with 'from' and 'to' junctions
            from_j = edge_static['from']
            to_j = edge_static['to']

            # Get global node indices for the junctions
            from_idx = static_junction_ids_to_index[from_j] + junction_offset
            to_idx = static_junction_ids_to_index[to_j] + junction_offset

            # Sort vehicles on this edge by their position (ascending)
            veh_indices_sorted = sorted(
                veh_indices,
                key=lambda i: current_vehicle_position_on_edges[i].item()
            )

            # Build dynamic edges along this edge
            for i, veh_idx in enumerate(veh_indices_sorted):
                # Compute global index of the vehicle node (offset by num_junctions)
                veh_global_idx = len(static_junction_ids_to_index) + veh_idx

                if i == 0:
                    # First vehicle on the edge → connect from start junction
                    dynamic_edge_index[0].append(from_idx)
                    dynamic_edge_index[1].append(veh_global_idx)
                    dynamic_edge_type.append(1)  # Type 1: JUNCTION → VEHICLE
                else:
                    # Connect from the previous vehicle on this edge
                    prev_veh_idx = veh_indices_sorted[i - 1]
                    prev_veh_global_idx = len(static_junction_ids_to_index) + prev_veh_idx
                    dynamic_edge_index[0].append(prev_veh_global_idx)
                    dynamic_edge_index[1].append(veh_global_idx)
                    dynamic_edge_type.append(3)  # Type 3: VEHICLE → VEHICLE
                # All dynamic edges get dummy edge features (currently zeros)
                dynamic_edge_attr.append([0.0] * edge_feature_dim)

                if i == len(veh_indices_sorted) - 1:
                    # Last vehicle on the edge → connect to end junction
                    dynamic_edge_index[0].append(veh_global_idx)
                    dynamic_edge_index[1].append(to_idx)
                    dynamic_edge_type.append(2)  # Type 2: VEHICLE → JUNCTION
                    # All dynamic edges get dummy edge features (currently zeros)
                    dynamic_edge_attr.append([0.0] * edge_feature_dim)

               
        if len(dynamic_edge_index[0]) != len(dynamic_edge_index[1]) or len(dynamic_edge_type) != len(dynamic_edge_index[0]):
            print(f"ERROR: Dynamic edge index and type lengths do not match. Terminating.")
            exit(1)
        if len(dynamic_edge_attr) != len(dynamic_edge_index[0]):
            print(f"ERROR: Dynamic edge attributes length {len(dynamic_edge_attr)} does not match edge index length {len(dynamic_edge_index[0])}. Terminating.")
            exit(1)

        return dynamic_edge_index, dynamic_edge_type, dynamic_edge_attr

    def validate_snapshots_and_labels(self):
        """
        Validates that snapshot and label data correspond correctly.
        
        This function goes through each pair of snapshot and label files and validates:
        1. File existence and basic structure
        2. Vehicle ID consistency between snapshots and labels
        3. Step number correspondence
        4. Data integrity and completeness
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        print("🔍 Starting snapshot and label validation...")
        
        validation_results = {
            'total_pairs': 0,
            'valid_pairs': 0,
            'invalid_pairs': 0,
            'missing_files': [],
            'vehicle_id_mismatches': [],
            'step_number_mismatches': [],
            'data_integrity_issues': [],
            'empty_snapshots': [],
            'empty_labels': []
        }
        
        for snap_file in tqdm(self.snapshot_files, desc="Validating snapshot-label pairs"):
            validation_results['total_pairs'] += 1
            
            # Extract step number from snapshot filename
            step_number = extract_step_number(snap_file)
            if step_number == -1:
                validation_results['data_integrity_issues'].append({
                    'file': snap_file,
                    'issue': 'Invalid step number in filename'
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Construct corresponding label filename
            label_file = snap_file.replace("step_", "labels_")
            label_path = os.path.join(self.labels_folder, label_file)
            snap_path = os.path.join(self.snapshots_folder, snap_file)
            
            # Check file existence
            if not os.path.exists(snap_path):
                validation_results['missing_files'].append({
                    'type': 'snapshot',
                    'file': snap_path,
                    'expected_label': label_path
                })
                validation_results['invalid_pairs'] += 1
                continue
                
            if not os.path.exists(label_path):
                validation_results['missing_files'].append({
                    'type': 'label',
                    'file': label_path,
                    'expected_snapshot': snap_path
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Load and validate snapshot data
            try:
                with open(snap_path, 'r') as f:
                    snapshot_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                validation_results['data_integrity_issues'].append({
                    'file': snap_path,
                    'issue': f'Failed to load snapshot JSON: {str(e)}'
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Load and validate label data
            try:
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                validation_results['data_integrity_issues'].append({
                    'file': label_path,
                    'issue': f'Failed to load label JSON: {str(e)}'
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Validate snapshot structure
            if not isinstance(snapshot_data, dict):
                validation_results['data_integrity_issues'].append({
                    'file': snap_path,
                    'issue': 'Snapshot data is not a dictionary'
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Check snapshot step number
            snapshot_step = snapshot_data.get('step')
            if snapshot_step != step_number:
                validation_results['step_number_mismatches'].append({
                    'snapshot_file': snap_file,
                    'filename_step': step_number,
                    'data_step': snapshot_step
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Extract vehicle nodes from snapshot
            nodes = snapshot_data.get('nodes', [])
            if not nodes:
                validation_results['empty_snapshots'].append({
                    'file': snap_file,
                    'step': step_number
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            vehicle_nodes = [node for node in nodes if node.get('node_type') == 1]
            snapshot_vehicle_ids = {node.get('id') for node in vehicle_nodes}
            
            # Validate label structure
            if not isinstance(label_data, list):
                validation_results['data_integrity_issues'].append({
                    'file': label_path,
                    'issue': 'Label data is not a list'
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            if not label_data:
                validation_results['empty_labels'].append({
                    'file': label_file,
                    'step': step_number
                })
                validation_results['invalid_pairs'] += 1
                continue
            
            # Extract vehicle IDs from labels
            label_vehicle_ids = set()
            for label_entry in label_data:
                if not isinstance(label_entry, dict):
                    validation_results['data_integrity_issues'].append({
                        'file': label_path,
                        'issue': f'Label entry is not a dictionary: {label_entry}'
                    })
                    validation_results['invalid_pairs'] += 1
                    break
                
                vehicle_id = label_entry.get('vehicle_id')
                eta = label_entry.get('eta')
                
                if vehicle_id is None:
                    validation_results['data_integrity_issues'].append({
                        'file': label_path,
                        'issue': f'Label entry missing vehicle_id: {label_entry}'
                    })
                    validation_results['invalid_pairs'] += 1
                    break
                
                if eta is None:
                    validation_results['data_integrity_issues'].append({
                        'file': label_path,
                        'issue': f'Label entry missing eta: {label_entry}'
                    })
                    validation_results['invalid_pairs'] += 1
                    break
                
                if not isinstance(eta, (int, float)):
                    validation_results['data_integrity_issues'].append({
                        'file': label_path,
                        'issue': f'Label entry eta is not numeric: {eta}'
                    })
                    validation_results['invalid_pairs'] += 1
                    break
                
                label_vehicle_ids.add(vehicle_id)
            else:  # Only execute if no break occurred
                # Check vehicle ID correspondence
                if snapshot_vehicle_ids != label_vehicle_ids:
                    missing_in_labels = snapshot_vehicle_ids - label_vehicle_ids
                    missing_in_snapshot = label_vehicle_ids - snapshot_vehicle_ids
                    
                    validation_results['vehicle_id_mismatches'].append({
                        'step': step_number,
                        'snapshot_file': snap_file,
                        'label_file': label_file,
                        'snapshot_vehicles': len(snapshot_vehicle_ids),
                        'label_vehicles': len(label_vehicle_ids),
                        'missing_in_labels': list(missing_in_labels),
                        'missing_in_snapshot': list(missing_in_snapshot)
                    })
                    validation_results['invalid_pairs'] += 1
                    continue
                
                # Additional validation: check vehicle node data integrity
                for vehicle_node in vehicle_nodes:
                    vehicle_id = vehicle_node.get('id')
                    required_fields = ['node_type', 'vehicle_type', 'speed', 'acceleration', 
                                     'current_edge', 'current_position', 'current_zone']
                    
                    for field in required_fields:
                        if field not in vehicle_node:
                            validation_results['data_integrity_issues'].append({
                                'file': snap_path,
                                'issue': f'Vehicle {vehicle_id} missing required field: {field}'
                            })
                            validation_results['invalid_pairs'] += 1
                            break
                    else:
                        continue
                    break
                else:  # Only execute if no break occurred in the inner loop
                    # All validations passed
                    validation_results['valid_pairs'] += 1
        
        # Print validation summary
        print(f"\n📊 Validation Summary:")
        print(f"   Total pairs checked: {validation_results['total_pairs']}")
        print(f"   Valid pairs: {validation_results['valid_pairs']}")
        print(f"   Invalid pairs: {validation_results['invalid_pairs']}")
        print(f"   Success rate: {validation_results['valid_pairs']/validation_results['total_pairs']*100:.1f}%")
        
        if validation_results['missing_files']:
            print(f"\n❌ Missing files ({len(validation_results['missing_files'])}):")
            for missing in validation_results['missing_files'][:5]:  # Show first 5
                print(f"   - {missing['type']}: {missing['file']}")
            if len(validation_results['missing_files']) > 5:
                print(f"   ... and {len(validation_results['missing_files']) - 5} more")
        
        if validation_results['vehicle_id_mismatches']:
            print(f"\n⚠️  Vehicle ID mismatches ({len(validation_results['vehicle_id_mismatches'])}):")
            for mismatch in validation_results['vehicle_id_mismatches'][:3]:  # Show first 3
                print(f"   - Step {mismatch['step']}: {mismatch['snapshot_vehicles']} vs {mismatch['label_vehicles']} vehicles")
            if len(validation_results['vehicle_id_mismatches']) > 3:
                print(f"   ... and {len(validation_results['vehicle_id_mismatches']) - 3} more")
        
        if validation_results['step_number_mismatches']:
            print(f"\n⚠️  Step number mismatches ({len(validation_results['step_number_mismatches'])}):")
            for mismatch in validation_results['step_number_mismatches'][:3]:  # Show first 3
                print(f"   - {mismatch['snapshot_file']}: filename={mismatch['filename_step']}, data={mismatch['data_step']}")
            if len(validation_results['step_number_mismatches']) > 3:
                print(f"   ... and {len(validation_results['step_number_mismatches']) - 3} more")
        
        if validation_results['empty_snapshots']:
            print(f"\n⚠️  Empty snapshots ({len(validation_results['empty_snapshots'])}):")
            for empty in validation_results['empty_snapshots'][:3]:  # Show first 3
                print(f"   - {empty['file']} (step {empty['step']})")
            if len(validation_results['empty_snapshots']) > 3:
                print(f"   ... and {len(validation_results['empty_snapshots']) - 3} more")
        
        if validation_results['empty_labels']:
            print(f"\n⚠️  Empty labels ({len(validation_results['empty_labels'])}):")
            for empty in validation_results['empty_labels'][:3]:  # Show first 3
                print(f"   - {empty['file']} (step {empty['step']})")
            if len(validation_results['empty_labels']) > 3:
                print(f"   ... and {len(validation_results['empty_labels']) - 3} more")
        
        if validation_results['data_integrity_issues']:
            print(f"\n❌ Data integrity issues ({len(validation_results['data_integrity_issues'])}):")
            for issue in validation_results['data_integrity_issues'][:3]:  # Show first 3
                print(f"   - {issue['file']}: {issue['issue']}")
            if len(validation_results['data_integrity_issues']) > 3:
                print(f"   ... and {len(validation_results['data_integrity_issues']) - 3} more")
        
        if validation_results['invalid_pairs'] > 0:
            print(f"\n❌ Validation failed! {validation_results['invalid_pairs']} pairs have issues.")
            return False
        else:
            print(f"\n✅ All snapshot-label pairs are valid!")
            return True

    def fast_validate_snapshots_and_labels(self, sample_size=1000):
        print("\n🔍 Fast validating snapshot-label pairs (sampled)...")
        pairs = list(zip(self.snapshot_files, [f.replace('step_', 'labels_') for f in self.snapshot_files]))
        if len(pairs) > sample_size:
            pairs = random.sample(pairs, sample_size)
        mismatches = 0
        for snap_file, label_file in tqdm(pairs, desc="Validating pairs", unit="pair"):
            snap_path = os.path.join(self.snapshots_folder, snap_file)
            label_path = os.path.join(self.labels_folder, label_file)
            if not os.path.exists(snap_path) or not os.path.exists(label_path):
                print(f"Missing file: {snap_path if not os.path.exists(snap_path) else label_path}")
                mismatches += 1
                continue
            try:
                with open(snap_path, 'r') as f:
                    snap_data = json.load(f)
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
            except Exception as e:
                print(f"Error loading files: {snap_file}, {label_file}: {e}")
                mismatches += 1
                continue
            vehicle_nodes = [node for node in snap_data.get('nodes', []) if node.get('node_type') == 1]
            vehicle_ids = set(node.get('id') for node in vehicle_nodes)
            if len(vehicle_nodes) != len(label_data):
                print(f"Count mismatch: {snap_file} ({len(vehicle_nodes)} vehicles) vs {label_file} ({len(label_data)} labels)")
                mismatches += 1
                continue
            # Check vehicle_id match and eta calculation
            snap_step = snap_data.get('step', None)
            for label in label_data:
                vid = label.get('vehicle_id')
                if vid not in vehicle_ids:
                    print(f"Vehicle ID {vid} in label not found in snapshot {snap_file}")
                    mismatches += 1
                    break
                # Validate ETA calculation if possible
                dest = label.get('destination_time_sec')
                eta = label.get('eta')
                if snap_step is not None and dest is not None and eta is not None:
                    expected_eta = max(dest - snap_step, 0)
                    if eta != expected_eta:
                        print(f"ETA mismatch for vehicle {vid} in {label_file}: label eta={eta}, expected {expected_eta} (dest={dest}, step={snap_step})")
                        mismatches += 1
                        break
        print(f"\nFast validation complete. {len(pairs)} pairs checked, {mismatches} mismatches found.")
        if mismatches > 0:
            print("❌ Fast validation failed!")
            return False
        print("✅ Fast validation passed!")
        return True

    def create_dataset(self):
        # Validate snapshot and label pairs before processing (unless skipped)
        if not self.skip_validation:
            if self.fast_validation:
                if not self.fast_validate_snapshots_and_labels():
                    print("❌ Dataset creation aborted due to fast validation failures.")
                    return
            else:
                if not self.validate_snapshots_and_labels():
                    print("❌ Dataset creation aborted due to validation failures.")
                    return
        else:
            print("⚠️  Skipping snapshot-label validation as requested.")
        
        print(f"Creating static junction data...")
        static_junction_ids_to_index, static_junction_features = self.process_junctions()
        
        # Validate junction features
        for i, features in enumerate(static_junction_features):
            if len(features) != NODE_FEATURES_COUNT:
                print(f"ERROR: Junction {i} has {len(features)} features, expected {NODE_FEATURES_COUNT}. Terminating.")
                exit(1)
        print(f"✓ Junction features validated: {len(static_junction_features)} junctions with {NODE_FEATURES_COUNT} features each")
        
        print("creating static edge data...")
        static_edge_index, static_edge_type, static_edge_ids_to_index, static_edge_features = self.process_static_edges(static_junction_ids_to_index)   
        
        # Validate edge features
        for i, features in enumerate(static_edge_features):
            if len(features) != EDGE_FEATURES_COUNT:
                print(f"ERROR: Edge {i} has {len(features)} features, expected {EDGE_FEATURES_COUNT}. Terminating.")
                exit(1)
        print(f"✓ Edge features validated: {len(static_edge_features)} edges with {EDGE_FEATURES_COUNT} features each")
        
        print(f"Converting {len(self.snapshot_files)} snapshots to PyG Data objects...")
        for snap_file in tqdm(self.snapshot_files, desc="Processing snapshots"):
            y_tensor, current_vehicle_ids, y_cat_tensors, y_binary_tensor = self.process_labels(snap_file)
            
            # Skip if no vehicles in this snapshot
            if len(current_vehicle_ids) == 0:
                print(f"WARNING: Snapshot {snap_file} has no vehicles. Skipping.")
                continue
                
            snapshot_data, vehicle_features, vehicle_routes_flat, vehicle_route_splits, current_vehicle_current_edges, current_vehicle_position_on_edges  = self.process_vehicle_features(snap_file, current_vehicle_ids, static_edge_ids_to_index, static_edge_features)
            
            # Validate vehicle features
            for i, features in enumerate(vehicle_features):
                if len(features) != NODE_FEATURES_COUNT:
                    print(f"ERROR: Vehicle {current_vehicle_ids[i]} has {len(features)} features, expected {NODE_FEATURES_COUNT}. Terminating.")
                    exit(1)
            
            vehicle_routes_flat_tensor = torch.LongTensor(vehicle_routes_flat)
            vehicle_route_splits_tensor = torch.LongTensor(vehicle_route_splits)
            current_vehicle_current_edges_tensor = torch.LongTensor(current_vehicle_current_edges)
            current_vehicle_position_on_edges_tensor = torch.FloatTensor(current_vehicle_position_on_edges)

           
            if len(y_tensor) != len(current_vehicle_ids):
                print(f"ERROR: y_tensor length {len(y_tensor)} does not match current_vehicle_ids length {len(current_vehicle_ids)}. Terminating.")
                exit(1)
            if len(static_edge_index[0]) != len(static_edge_index[1]) or len(static_edge_type) != len(static_edge_index[0]):
                print(f"ERROR: static_edge_index and static_edge_type lengths do not match. Terminating.")
                exit(1)
            if len(static_edge_features) != len(static_edge_ids_to_index):
                print(f"ERROR: static_edge_features length {len(static_edge_features)} does not match static_edge_ids_to_index length {len(static_edge_ids_to_index)}. Terminating.")
                exit(1) 
            
            static_edge_features_updated = self.update_edge_features(snapshot_data, current_vehicle_ids, static_edge_features)
            edge_attr_tensor = torch.FloatTensor(static_edge_features_updated)
            
            # Validate updated edge features
            for i, features in enumerate(static_edge_features_updated):
                if len(features) != EDGE_FEATURES_COUNT:
                    print(f"ERROR: Updated edge {i} has {len(features)} features, expected {EDGE_FEATURES_COUNT}. Terminating.")
                    exit(1)

            # 1. get vehicle features
            # 2. for each vehicle, update edge demand and occupancy from static_edge_features_updated

            for idx, veh_features in enumerate(vehicle_features):
                veh_features[23] = static_edge_features_updated[veh_features[22]][5]
                veh_features[24] = static_edge_features_updated[veh_features[22]][6]


            x = [*static_junction_features, *vehicle_features]  # Combine static junction features with vehicle features
            x_tensor = torch.FloatTensor(x) # nodes features tensor = [static junction features, vehicle features]
 
            if len(x_tensor) != len(static_junction_features) + len(current_vehicle_ids):
                print(f"ERROR: x_tensor length {len(x_tensor)} does not match expected length {len(static_junction_features) + len(current_vehicle_ids)}. Terminating.")
                exit(1)

            dynamic_edge_index, dynamic_edge_type, dynamic_edge_attr = self.construct_dynamic_edges(
                current_vehicle_ids=current_vehicle_ids,
                current_vehicle_current_edges=current_vehicle_current_edges_tensor,
                current_vehicle_position_on_edges=current_vehicle_position_on_edges_tensor,
                static_edge_ids_to_index=static_edge_ids_to_index,
                static_junction_ids_to_index=static_junction_ids_to_index,
                static_edge_features=static_edge_features_updated,
                static_edge_info=self.entities_data['edge']['features'],
                edge_feature_dim=edge_attr_tensor.shape[1],
                junction_offset=0
            )

            full_edge_index = [
                static_edge_index[0] + dynamic_edge_index[0],
                static_edge_index[1] + dynamic_edge_index[1]
            ]
            full_edge_type = static_edge_type + dynamic_edge_type
            full_edge_attr_tensor = torch.cat([edge_attr_tensor, torch.FloatTensor(dynamic_edge_attr)], dim=0)

            edge_index_tensor = torch.tensor(full_edge_index, dtype=torch.long)
            edge_type_tensor = torch.tensor(full_edge_type, dtype=torch.long)

            if edge_index_tensor.shape[1] != len(full_edge_type):
                print(f"ERROR: Edge index tensor shape {edge_index_tensor.shape} does not match edge type length {len(full_edge_type)}. Terminating.")
                exit(1)
            if edge_index_tensor.shape[0] != 2:
                print(f"ERROR: Edge index tensor should have 2 rows, found {edge_index_tensor.shape[0]}. Terminating.")
                exit(1)
            if edge_type_tensor.shape[0] != len(full_edge_type):
                print(f"ERROR: Edge type tensor length {edge_type_tensor.shape[0]} does not match full_edge_type length {len(full_edge_type)}. Terminating.")
                exit(1)
            if edge_attr_tensor.shape[0] != len(static_edge_ids_to_index):
                print(f"ERROR: Edge attribute tensor length {edge_attr_tensor.shape[0]} does not match static_edge_ids_to_index length {len(static_edge_ids_to_index)}. Terminating.")
                exit(1)
            if edge_attr_tensor.shape[1] != full_edge_attr_tensor.shape[1]:
                print(f"ERROR: Edge attribute tensor feature dimension {edge_attr_tensor.shape[1]} does not match full_edge_attr_tensor feature dimension {full_edge_attr_tensor.shape[1]}. Terminating.")
                exit(1)
            if full_edge_index[0] is None or full_edge_index[1] is None:
                print(f"ERROR: Full edge index has None values. Terminating.")
                exit(1)
            if len(full_edge_index[0]) != len(full_edge_index[1]) or len(full_edge_type) != len(full_edge_index[0]):
                print(f"ERROR: Full edge index and type lengths do not match. Terminating.")
                exit(1)
            


            data = Data(
                        x=x_tensor,
                        edge_index=edge_index_tensor,
                        edge_type=edge_type_tensor,
                        edge_attr=full_edge_attr_tensor,
                        vehicle_ids=current_vehicle_ids,
                        junction_ids=list(static_junction_ids_to_index.keys()),
                        edge_ids=list(static_edge_ids_to_index.keys()),
                        vehicle_routes=vehicle_routes_flat_tensor,
                        vehicle_route_splits=vehicle_route_splits_tensor,
                        current_vehicle_current_edges=current_vehicle_current_edges_tensor,
                        current_vehicle_position_on_edges=current_vehicle_position_on_edges_tensor,
                        y=y_tensor,
                        y_equal_thirds=y_cat_tensors['equal_thirds'],
                        y_quartile=y_cat_tensors['quartile'],
                        y_mean_pm_0_5_std=y_cat_tensors['mean_pm_0_5_std'],
                        y_median_pm_0_5_iqr=y_cat_tensors['median_pm_0_5_iqr'],
                        y_binary_eta=y_binary_tensor
            )
            
            # Final validation of the PyG Data object
            if data.x.shape[1] != NODE_FEATURES_COUNT:
                print(f"ERROR: Final node features have {data.x.shape[1]} dimensions, expected {NODE_FEATURES_COUNT}. Terminating.")
                exit(1)
            if data.edge_attr.shape[1] != EDGE_FEATURES_COUNT:
                print(f"ERROR: Final edge features have {data.edge_attr.shape[1]} dimensions, expected {EDGE_FEATURES_COUNT}. Terminating.")
                exit(1)
            
            # Save the data object to a .pt file
            out_file = os.path.join(self.out_graph_folder, snap_file.replace(".json", ".pt"))
            torch.save(data, out_file)
            # print(f"✓ Saved {snap_file.replace('.json', '.pt')} with {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")


def main():
    parser = argparse.ArgumentParser(description="Create a traffic dataset from simulation snapshots.")
    parser.add_argument(
        '--config', 
        default="/home/guy/Projects/Traffic/Traffic-DSTG-Gen/simulation.config.json", 
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
        default="/home/guy/Projects/Traffic/Traffic-DSTG-Gen/eda_exports",
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
        "--normalize",
        action="store_true",
        default=True,
        help="Enable min-max normalization of feature and labels (default: True)"
    )

    parser.add_argument(
        "--log_normalize",
        action="store_true",
        default=True,
        help="Enable log + z-score normalization of features (default: True)"
    )

    parser.add_argument(
        "--mapping_folder",
        type=str,
        default="/home/guy/Projects/Traffic/Traffic-DSTG-Gen/eda_exports/mappings",
        help="Folder with mappings for vehicle, junction, and edge features (default: eda_exports/mappings)"
    )
    
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        default=True,
        help="Skip snapshot-label validation (default: False)"
    )

    parser.add_argument(
        "--eta_analysis_methods_path",
        type=str,
        default="/home/guy/Projects/Traffic/Traffic-DSTG-Gen/eda_exports/eta/eta_analysis_methods.csv",
        help="Path to eta_analysis_methods.csv file (default: eda_exports/eta/eta_analysis_methods.csv)"
    )

    parser.add_argument(
        "--fast_validation",
        action="store_true",
        default=True,
        help="Enable fast validation by sampling 1000 snapshot-label pairs."
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
        args.normalize,
        args.log_normalize,
        args.mapping_folder,
        args.eta_analysis_methods_path,
        args.skip_validation,
        args.fast_validation
        )
    creator.create_dataset()


if __name__ == "__main__":
    main()