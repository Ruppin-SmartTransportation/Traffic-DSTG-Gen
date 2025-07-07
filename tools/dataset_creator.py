'''
x - Nodes (Junctions, Vehicle) main feature layout:

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
import re

def extract_step_number(filename):
    match = re.search(r"step_(\d+)\.json", filename)
    return int(match.group(1)) if match else -1

def extract_numeric_suffix(s):
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else float('inf')

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
        
        if not self.check_required_files():
            raise FileNotFoundError("Required files for dataset creation are missing. Please ensure all necessary files are present in the specified directories.")
        with open(config) as f:
            self.config = json.load(f)
        self.filter_outliers = filter_outliers
        self.normalize = z_normalize
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
                if entity == 'vehicle':
                    # Sort vehicle IDs by numeric suffix to ensure consistent order
                    self.entities_data[entity]['ids'] = sorted(
                        data_map.keys(),
                        key=extract_numeric_suffix
                    )
                else:
                    # For junctions and edges, we can sort by keys directly
                    self.entities_data[entity]['ids'] = sorted(data_map.keys())
                print(f"entity {entity} has {len(self.entities_data[entity]['ids'])} ids")
                print(f"self.entities_data['{entity}']['ids'][:5]", self.entities_data[entity]['ids'][:5])
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
                print(f"[{entity}]['stats'][{feature_name}]", self.entities_data[entity]['stats'][feature_name])

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

    def process_junctions(self):
        '''
        entity junction has 362 ids
        ['AA0', 'AA1', 'AA10', 'AA2', 'AA3']
        [junction]['stats'][id] {'keys': []}
        [junction]['stats'][node_type] {'keys': [0]}
        [junction]['stats'][x] {'mean': 8944.463176795582, 'std': 5522.114834273921, 'min': 0.0, 'max': 18000.0}
        [junction]['stats'][y] {'mean': 405.8405524861883, 'std': 3351.537530194094, 'min': -6264.96, 'max': 5000.0}
        [junction]['stats'][type] {'keys': ['priority', 'traffic_light']}
        [junction]['stats'][zone] {'keys': ['A', 'B', 'C', 'H']}

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
        | 11    | `route_length_left` | z normalized                                            |
        | 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
        | 16    | `current_x`         | z normalized                                            |
        | 17    | `current_y`         | z normalized                                            |
        | 18    | `j_type`            | junction type, "priority", "traffic_light"              |



        '''
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
            features = [0] * 12  # Initialize with zeros
            # junction node feature are node_type (0), zone (one hot) x, y, j_type
            features[0] = 0  # node_type = 0 for junction
            zone_oh = [0] * len(stats['zone']['keys'])  # One-hot encoding for zone
            zone = static_features[jid].get('zone', '_')  # Default to '_' if not found
            if zone not in stats['zone']['keys']:
                print(f"ERROR: Junction {jid} has unknown zone '{zone}'. Terminating.")
                exit(1)
            zone_index = stats['zone']['keys'].index(zone)
            zone_oh[zone_index] = 1  # Set the correct zone index to
            features.extend(zone_oh)  # Add one-hot zone features
            x = static_features[jid].get('x', 0.0)
            y = static_features[jid].get('y', 0.0)
            if x is None or y is None:
                print(f"ERROR: Junction {jid} has no x or y defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                x = (x - stats['x']['min']) / (stats['x']['max'] - stats['x']['min'])
                y = (y - stats['y']['min']) / (stats['y']['max'] - stats['y']['min'])
           
            features.append(x)
            features.append(y)
            j_type = static_features[jid].get('type', 'unknown')
            if j_type not in stats['type']['keys']:
                print(f"ERROR: Junction {jid} has unknown type '{j_type}'. Terminating.")
                exit(1)
            features.append(stats['type']['keys'].index(j_type))  # Convert type to index
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
        | 0     | `avg_speed`            | z normalized                                         |
        | 1-3   | `num_lanes`            | One-hot of 1-3 lanes                                 |
        | 4     | `length`               | z normalized (distance between from and to nodes)    |
        | 5     | `edge_demand`          | log + z-score normalized                             |
        | 6     | `edge_occupancy`       | z normalized                                         |

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
                print(f"ERROR: Edge {eid} has unknown num_lanes '{num_lanes}'. Terminating.")
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
                length = (length - stats['length']['min']) / (stats['length']['max'] - stats['length']['min'])
            features.append(length)
            # edge_demand and edge_occupancy will be calculated dynamically per snapshot
            features.append(0)  # Placeholder for edge_demand
            features.append(0)  # Placeholder for edge_occupancy
            assert len(features) == 7, f"Edge features for {eid} should have 7 elements, got {len(features)}"
            static_edge_features.append(features)       
        
        return  static_edge_index, static_edge_type, static_edge_ids_to_index, static_edge_features

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
        mean_eta = eta_stats.get("mean", 0.0)
        std_eta = eta_stats.get("std", 1.0)
        
        label_map = {}
        valid_vehicle_ids = set()
        if std_eta == 0.0:
            std_eta = 1.0
        filter_outliers_count = 0
        for entry in label_data:
            eta = entry.get('eta')
            vid = entry.get('vehicle_id')
            if eta is None or vid is None:
                print(f"Warning: Missing 'eta' or 'vehicle_id' in label entry {entry}. Terminating.")
                exit(1)

            if self.normalize :
                z = (eta - mean_eta) / std_eta

                if self.filter_outliers and abs(z) > 4:
                    filter_outliers_count += 1
                    print(f"Filtering out vehicle {vid} with outlier ETA z-score: {z:.2f} etsa: {eta:.2f}")
                    continue  # Outlier → skip

                eta_final = z # Use z-score normalized value
            else:
                eta_final = eta

            label_map[vid] = eta_final
            valid_vehicle_ids.add(vid)
        if len(label_map) != len(valid_vehicle_ids):
            print(f"Warning: Mismatch in label_map and valid_vehicle_ids length. {len(label_map)} vs {len(valid_vehicle_ids)}")
            exit(1)
        # sort the vehicle IDs to ensure consistent order
        valid_vehicle_ids = sorted(valid_vehicle_ids, key=extract_numeric_suffix)
        etas = [label_map[vid] for vid in valid_vehicle_ids]
        y = torch.FloatTensor(etas)

        return y, valid_vehicle_ids

    def process_vehicle_features(self, snap_file, current_vehicle_ids):
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
        | 11    | `route_length_left` | z normalized                                            |
        | 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
        | 16    | `current_x`         | z normalized                                            |
        | 17    | `current_y`         | z normalized                                            |
        | 18    | `j_type`            | junction type, "priority", "traffic_light"              |

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
        for vid in current_vehicle_ids:
            vehicle_node = next((node for node in vehicle_nodes if node.get('id') == vid), None)
            if vehicle_node is None:
                print(f"ERROR: Vehicle {vid} not found in snapshot {snap_file}. Terminating.")
                exit(1)
            feature = []  # Initialize feature vector
            # node_type = 1 for vehicle
            if 'node_type' not in vehicle_node:
                print(f"ERROR: Vehicle {vid} node does not have 'node_type' defined. Terminating.")
                exit(1)
            if vehicle_node['node_type'] != 1:
                print(f"ERROR: Vehicle {vid} node_type is not 1 (vehicle). Found: {vehicle_node['node_type']}. Terminating.")
                exit(1)
            feature.append(1)  # node_type = 1 for vehicle
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
            feature.extend(veh_type_oh)  # Add one-hot vehicle type features
            # speed
            speed = vehicle_node.get('speed', None)
            if speed is None:
                print(f"ERROR: Vehicle {vid} has no speed defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                speed = (speed - vehicle_stats['speed']['min']) / (vehicle_stats['speed']['max'] - vehicle_stats['speed']['min'])
            feature.append(speed)
            # acceleration
            acceleration = vehicle_node.get('acceleration', None)
            if acceleration is None:
                print(f"ERROR: Vehicle {vid} has no acceleration defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                acceleration = (acceleration - vehicle_stats['acceleration']['min']) / (vehicle_stats['acceleration']['max'] - vehicle_stats['acceleration']['min'])   
            feature.append(acceleration)
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
            feature.append(math.sin(2 * math.pi * hour / 24))  # sin_hour
            feature.append(math.cos(2 * math.pi * hour / 24))  # cos_hour
            feature.append(math.sin(2 * math.pi * day / 7))  # sin_day
            feature.append(math.cos(2 * math.pi * day / 7))  # cos_day
            # route_length 
            route_length = vehicle_node.get('route_length', None)
            if route_length is None:
                print(f"ERROR: Vehicle {vid} has no route_length defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                route_length = (route_length - vehicle_stats['route_length']['min']) / (vehicle_stats['route_length']['max'] - vehicle_stats['route_length']['min'])
            feature.append(route_length)
            # route_length_left
            route_length_left = vehicle_node.get('route_length_left', None)
            if route_length_left is None:
                print(f"ERROR: Vehicle {vid} has no route_length_left defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                route_length_left = (route_length_left - vehicle_stats['route_length_left']['min']) / (vehicle_stats['route_length_left']['max'] - vehicle_stats['route_length_left']['min'])
            feature.append(route_length_left)
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
            feature.extend(zone_oh)  # Add one-hot zone features
            # current_x, current_y
            current_x = vehicle_node.get('current_x', None)
            current_y = vehicle_node.get('current_y', None)
            if current_x is None or current_y is None:
                print(f"ERROR: Vehicle {vid} has no current_x or current_y defined. Terminating.")
                exit(1)
            if self.normalize: # use min-max normalization instead of z-score
                
                current_x = (current_x - vehicle_stats['current_x']['min']) / (vehicle_stats['current_x']['max'] - vehicle_stats['current_x']['min'])
                current_y = (current_y - vehicle_stats['current_y']['min']) / (vehicle_stats['current_y']['max'] - vehicle_stats['current_y']['min'])
            feature.append(current_x)
            feature.append(current_y)
            # j_type is the junction type, not relevant for vehicles, but we need to add it to the feature vector set to 0
            feature.append(0)  # j_type = 0 for vehicle (not relevant)
            # Add the feature vector to the list
            features.append(feature)
            
        # Convert features to a tensor
        x = torch.FloatTensor(features)
        return x, snapshot_data

    def update_edge_features(self, snapshot_data, current_vehicle_ids, static_edge_features):

        """
        Updates edge features based on the current vehicle and dynamic attributes.
        | Index | Feature Name        | Notes                                                   |

        | ----- | ---------------------- | ---------------------------------------------------- |
        | 0     | `avg_speed`            | z normalized                                         |
        | 5     | `edge_demand`          | log + z-score normalized                             |
        | 6     | `edge_occupancy`       | z normalized                                         |

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
            route = vehicle_node.get('route', [])
            if not route:
                continue
            # add 1 for each edge in the route of the vehicle
            for edge_id in route:
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

            # Calculate edge_demand based on the edge_demand_map
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
            
            # Calculate edge_occupancy based on number of vehicles on the edge
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
           
    def create_dataset(self):
        print(f"Creating static junction data...")
        static_junction_ids_to_index, static_junction_features = self.process_junctions()
        print("creating static edge data...")
        static_edge_index, static_edge_type, static_edge_ids_to_index, static_edge_features = self.process_static_edges(static_junction_ids_to_index)   
        print(f"Converting {len(self.snapshot_files)} snapshots to PyG Data objects...")
        for snap_file in tqdm(self.snapshot_files, desc="Processing snapshots"):
            y_tensor, current_vehicle_ids = self.process_labels(snap_file)
            vehicles_x, snapshot_data = self.process_vehicle_features(snap_file, current_vehicle_ids)
            x = [*static_junction_features, *vehicles_x]  # Combine static junction features with vehicle features
            x_tensor = torch.FloatTensor(x)
            
            # print(f"Processing snapshot {snap_file} with {len(current_vehicle_ids)} vehicles...")
            static_edge_features_updated = self.update_edge_features(snapshot_data, current_vehicle_ids, static_edge_features)
            edge_attr_tensor = torch.FloatTensor(static_edge_features_updated)
            
            edge_index = torch.tensor(static_edge_index, dtype=torch.long)
            edge_type = torch.tensor(static_edge_type, dtype=torch.long)
            data = Data(
                        x=x_tensor, 
                        edge_index=edge_index, edge_type=edge_type, y=y_tensor, 
                        edge_attr=edge_attr_tensor,
                        static_junction_ids_to_index=static_junction_ids_to_index,
                        static_edge_ids_to_index=static_edge_ids_to_index,
                        current_vehicle_ids=current_vehicle_ids)
            # Save the data object to a .pt file
            out_file = os.path.join(self.out_graph_folder, snap_file.replace(".json", ".pt"))
            torch.save(data, out_file)


            

            

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