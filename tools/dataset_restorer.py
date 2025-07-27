import torch
import json
import os
import numpy as np
import argparse
import pandas as pd
import re

def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class DatasetRestorer:
    """
    Restores a PyG .pt file (graph snapshot) to a human-readable JSON format by inverting normalization and decoding features.
    """
    def __init__(self, stats_dir, mapping_dir):
        """
        Args:
            stats_dir (str): Directory containing normalization statistics (CSVs or JSONs).
            mapping_dir (str): Directory containing mapping files for vehicle, junction, and edge IDs.
        """
        self.stats_dir = stats_dir
        self.mapping_dir = mapping_dir
        if not self.check_required_files():
            raise FileNotFoundError("Required files for dataset restoration are missing. Please ensure all necessary files are present in the specified directories.")
        self.stats = self.load_stats()
        self.mappings = self.load_mappings()

    def check_required_files(self):
        """
        Check for the presence of required stats and mapping files.
        Returns:
            bool: True if all required files are present, False otherwise.
        """
        required_files = [
            os.path.join(self.stats_dir, 'junction_feature_summary.csv'),
            os.path.join(self.stats_dir, 'vehicle_feature_summary.csv'),
            os.path.join(self.stats_dir, 'edge_feature_summary.csv'),
            os.path.join(self.stats_dir, 'labels_feature_summary.csv'),
            os.path.join(self.mapping_dir, 'vehicle_mapping.json'),
            os.path.join(self.mapping_dir, 'junction_mapping.json'),
            os.path.join(self.mapping_dir, 'edge_mapping.json'),
        ]
        all_present = True
        for file in required_files:
            if not os.path.exists(file):
                print(f"Missing required file: {file}")
                all_present = False
        if all_present:
            print("All required files are present.")
        return all_present

    def load_stats(self):
        """
        Load normalization statistics from summary CSVs in stats_dir.
        Returns:
            dict: Nested dictionary of stats for each entity and feature.
        """
        entities = ['junction', 'vehicle', 'edge', 'label']
        stats_files = {
            'junction': os.path.join(self.stats_dir, 'junction_feature_summary.csv'),
            'vehicle': os.path.join(self.stats_dir, 'vehicle_feature_summary.csv'),
            'edge': os.path.join(self.stats_dir, 'edge_feature_summary.csv'),
            'label': os.path.join(self.stats_dir, 'labels_feature_summary.csv'),
        }
        stats = {}
        for entity in entities:
            stats[entity] = {}
            if not os.path.exists(stats_files[entity]):
                continue
            df = pd.read_csv(stats_files[entity])
            for _, row in df.iterrows():
                feature_name = row['feature']
                entry = {}
                if row['type'] == 'numeric':
                    entry['mean'] = float(row.get('mean', 0.0))
                    entry['std'] = float(row.get('std', 1.0))
                    entry['min'] = float(row.get('min', 0.0))
                    entry['max'] = float(row.get('max', 1.0))
                    for p in [97, 98, 99]:
                        percentile_name = f'{p}%'
                        if percentile_name in row:
                            entry[percentile_name] = float(row[percentile_name])
                elif row['type'] == 'categorical':
                    try:
                        value_counts = json.loads(row.get('value_counts', '{}').replace("'", '"'))
                    except Exception:
                        value_counts = {}
                    entry['keys'] = sorted(value_counts.keys()) if value_counts else []
                stats[entity][feature_name] = entry
        return stats

    def load_mappings(self):
        """
        Load mapping files for vehicle, junction, and edge IDs from mapping_dir.
        Returns:
            dict: Dictionary with keys 'vehicle', 'junction', 'edge' and their mappings.
        """
        mapping_files = {
            'vehicle': os.path.join(self.mapping_dir, 'vehicle_mapping.json'),
            'junction': os.path.join(self.mapping_dir, 'junction_mapping.json'),
            'edge': os.path.join(self.mapping_dir, 'edge_mapping.json'),
        }
        mappings = {}
        for entity, path in mapping_files.items():
            if os.path.exists(path):
                with open(path, 'r') as f:
                    mappings[entity] = json.load(f)
            else:
                mappings[entity] = {}
        return mappings

    def inverse_normalize(self, value, feature_name, entity_type):
        """
        Inverse normalization for a single value.
        Args:
            value (float): Normalized value.
            feature_name (str): Name of the feature.
            entity_type (str): 'node' or 'edge'.
        Returns:
            float: Original value.
        """
        stats = self.stats[entity_type][feature_name] if feature_name in self.stats[entity_type] else None
        if stats is None:
            return value
        if entity_type == 'edge' and feature_name == 'edge_demand':
            # Use edge_route_count_log stats for edge_demand
            log_stats = self.stats['edge'].get('edge_route_count_log', {})
            mean = log_stats.get('mean', 0.0)
            std = log_stats.get('std', 1.0)
            return np.exp(value + mean / std) - 1
        elif entity_type == 'edge' and feature_name == 'edge_occupancy':
            # Use vehicles_on_road_count_log stats for edge_occupancy
            log_stats = self.stats['edge'].get('vehicles_on_road_count_log', {})
            mean = log_stats.get('mean', 0.0)
            std = log_stats.get('std', 1.0)
            return np.exp(value + mean / std) - 1
        else:
            # Check if this is edge length which uses division by max normalization
            if entity_type == 'edge' and feature_name == 'length':
                # Edge length uses: length = length / stats['length']['max']
                # So inverse: length_orig = length * max
                max_v = stats.get('max', 1.0)
                return value * max_v
            else:
                # min-max inverse normalization: x_orig = x * (max - min) + min
                min_v = stats.get('min', 0.0)
                max_v = stats.get('max', 1.0)
                return value * (max_v - min_v) + min_v

    def decode_one_hot(self, one_hot_vec, keys):
        """
        Decode a one-hot vector to its original categorical value.
        Args:
            one_hot_vec (list or np.array): One-hot encoded vector.
            keys (list): List of possible categories.
        Returns:
            str: Decoded category.
        """
        idx = int(np.argmax(one_hot_vec))
        return keys[idx] if 0 <= idx < len(keys) else None

    def restore_junctions_from_pt(self, data):
        """
        Restore all junction nodes from the x tensor in the .pt file.
        Args:
            data: PyG Data object loaded from .pt file.
        Returns:
            list: List of restored junction entities (dicts).
        """
        junctions = []
        x = data.x.cpu().numpy()
        junction_ids = data.junction_ids if hasattr(data, 'junction_ids') else []
        stats = self.stats['junction']
        mappings = self.mappings['junction']
        for idx, node in enumerate(x):
            if node[0] != 0:
                continue  # Not a junction
            # Restore features (inverse normalization, decode one-hot, etc.)
            restored = {}
            # Example: restore zone
            zone_oh = node[12:16]
            zone_keys = stats['zone']['keys'] if 'zone' in stats else []
            restored['zone'] = self.decode_one_hot(zone_oh, zone_keys)
            # Restore x, y
            x_val = self.inverse_normalize(node[16], 'x', 'junction')
            y_val = self.inverse_normalize(node[17], 'y', 'junction')
            restored['x'] = x_val
            restored['y'] = y_val
            # Restore j_type (last feature)
            jtype_idx = int(node[-1])
            jtype_keys = stats['type']['keys'] if 'type' in stats else []
            restored['type'] = jtype_keys[jtype_idx] if 0 <= jtype_idx < len(jtype_keys) else None
            # Add ID if available
            if idx < len(junction_ids):
                restored['id'] = junction_ids[idx]
            junctions.append(restored)
        return junctions

    def restore_vehicles_from_pt(self, data):
        """
        Restore all vehicle nodes from the x tensor in the .pt file.
        Args:
            data: PyG Data object loaded from .pt file.
        Returns:
            list: List of restored vehicle entities (dicts).
        """
        vehicles = []
        x = data.x.cpu().numpy()
        vehicle_ids = data.vehicle_ids if hasattr(data, 'vehicle_ids') else []
        stats = self.stats['vehicle']
        mappings = self.mappings['vehicle']
        # Prepare route and position info if available
        edge_ids = data.edge_ids if hasattr(data, 'edge_ids') else []
        vehicle_routes = data.vehicle_routes.cpu().numpy() if hasattr(data, 'vehicle_routes') else None
        vehicle_route_splits = data.vehicle_route_splits.cpu().numpy() if hasattr(data, 'vehicle_route_splits') else None
        current_edges = data.current_vehicle_current_edges.cpu().numpy() if hasattr(data, 'current_vehicle_current_edges') else None
        current_positions = data.current_vehicle_position_on_edges.cpu().numpy() if hasattr(data, 'current_vehicle_position_on_edges') else None
        for idx, node in enumerate(x):
            if node[0] != 1:
                continue  # Not a vehicle
            restored = {}
            # Example: restore vehicle_type
            veh_type_oh = node[1:4]
            veh_type_keys = stats['vehicle_type']['keys'] if 'vehicle_type' in stats else []
            restored['vehicle_type'] = self.decode_one_hot(veh_type_oh, veh_type_keys)
            # Restore speed, acceleration
            restored['speed'] = self.inverse_normalize(node[4], 'speed', 'vehicle')
            restored['acceleration'] = self.inverse_normalize(node[5], 'acceleration', 'vehicle')
            # Restore time features
            restored['sin_hour'] = node[6]
            restored['cos_hour'] = node[7]
            restored['sin_day'] = node[8]
            restored['cos_day'] = node[9]
            # Restore route_length, progress
            restored['route_length'] = self.inverse_normalize(node[10], 'route_length', 'vehicle')
            restored['progress'] = node[11]
            # Restore zone
            zone_oh = node[12:16]
            zone_keys = stats['current_zone']['keys'] if 'current_zone' in stats else []
            restored['current_zone'] = self.decode_one_hot(zone_oh, zone_keys)
            # Restore current_x, current_y
            restored['current_x'] = self.inverse_normalize(node[16], 'current_x', 'vehicle')
            restored['current_y'] = self.inverse_normalize(node[17], 'current_y', 'vehicle')
            # Restore destination_x, destination_y
            restored['destination_x'] = self.inverse_normalize(node[18], 'destination_x', 'vehicle')
            restored['destination_y'] = self.inverse_normalize(node[19], 'destination_y', 'vehicle')
            # Restore current_edge_num_lanes (one-hot)
            num_lanes_oh = node[20:23]
            num_lanes_keys = stats['num_lanes']['keys'] if 'num_lanes' in stats else ['1', '2', '3']
            restored['current_edge_num_lanes'] = self.decode_one_hot(num_lanes_oh, num_lanes_keys)
            # Restore current_edge_demand, current_edge_occupancy (output as-is, skip inverse normalization)
            # These are left in their normalized form intentionally
            restored['current_edge_demand'] = node[23]
            restored['current_edge_occupancy'] = node[24]
            # j_type is always 0 for vehicles
            restored['j_type'] = 0
            # Add ID if available
            if idx - len(data.junction_ids) < len(vehicle_ids):
                restored['id'] = vehicle_ids[idx - len(data.junction_ids)]
            else:
                restored['id'] = -1
            # Restore route and route_left
            if vehicle_routes is not None and vehicle_route_splits is not None:
                # Calculate the vehicle index (accounting for junctions)
                vehicle_idx = len([n for n in x[:idx] if n[0] == 1])  # Count vehicles before this one
                if vehicle_idx < len(vehicle_route_splits):
                    route_len = vehicle_route_splits[vehicle_idx]
                    # Calculate route pointer for this vehicle
                    route_ptr = sum(vehicle_route_splits[:vehicle_idx])
                    route_edge_indices = vehicle_routes[route_ptr:route_ptr+route_len] if route_ptr + route_len <= len(vehicle_routes) else []
                    # Convert edge indices to edge IDs
                    route = [edge_ids[i] for i in route_edge_indices] if edge_ids is not None else []
                    restored['route'] = route
                else:
                    restored['route'] = []
                # For route_left, you may need to reconstruct based on progress or other info (skipped if not available)
            # Restore current_edge
            if current_edges is not None:
                vehicle_idx = len([n for n in x[:idx] if n[0] == 1])  # Count vehicles before this one
                if vehicle_idx < len(current_edges):
                    edge_idx = current_edges[vehicle_idx]
                    restored['current_edge'] = edge_ids[edge_idx] if edge_ids and edge_idx < len(edge_ids) else None
                else:
                    restored['current_edge'] = None
            # Restore current_position
            if current_positions is not None:
                vehicle_idx = len([n for n in x[:idx] if n[0] == 1])  # Count vehicles before this one
                if vehicle_idx < len(current_positions):
                    restored['current_position'] = float(current_positions[vehicle_idx])
                else:
                    restored['current_position'] = None
            vehicles.append(restored)
        return vehicles

    def restore_edges_from_pt(self, data):
        """
        Restore all edges from the .pt file (static edges only).
        Args:
            data: PyG Data object loaded from .pt file.
        Returns:
            list: List of restored edge entities (dicts).
        """
        edges = []
        if not hasattr(data, 'edge_attr') or not hasattr(data, 'edge_ids'):
            return edges
        edge_attrs = data.edge_attr.cpu().numpy()
        edge_ids = data.edge_ids
        stats = self.stats['edge']
        mappings = self.mappings['edge']
        # Only restore static edges (first len(edge_ids) entries)
        for idx, edge_id in enumerate(edge_ids):
            edge = {'id': edge_id}
            # Example: restore num_lanes (one-hot)
            num_lanes_oh = edge_attrs[idx][1:4]
            num_lanes_keys = stats.get('num_lanes', {}).get('keys', [])
            if not num_lanes_keys:  # If keys are empty, use default
                num_lanes_keys = ['1', '2', '3']
            decoded_lanes = self.decode_one_hot(num_lanes_oh, num_lanes_keys)
            edge['num_lanes'] = decoded_lanes
            # Restore avg_speed, length
            edge['avg_speed'] = self.inverse_normalize(edge_attrs[idx][0], 'avg_speed', 'edge')
            edge['length'] = self.inverse_normalize(edge_attrs[idx][4], 'length', 'edge')
            # Restore edge_demand, edge_occupancy (apply inverse normalization)
            edge['edge_demand'] = self.inverse_normalize(edge_attrs[idx][5], 'edge_demand', 'edge')
            edge['edge_occupancy'] = self.inverse_normalize(edge_attrs[idx][6], 'edge_occupancy', 'edge')
            # Optionally restore from/to if available in mapping
            if edge_id in mappings:
                if 'from' in mappings[edge_id]:
                    edge['from'] = mappings[edge_id]['from']
                if 'to' in mappings[edge_id]:
                    edge['to'] = mappings[edge_id]['to']
            edges.append(edge)
        return edges

    def check_missing_vehicle_durations(self, missing_vehicle_ids, labels_folder, snap_file):
        """
        Check travel duration values for missing vehicles in the corresponding label JSON file.
        
        Args:
            missing_vehicle_ids (list): List of vehicle IDs that are missing from restoration
            labels_folder (str): Path to the labels folder
            snap_file (str): Original snapshot filename (e.g., 'step_067260.json')
            
        Returns:
            dict: Results of duration checks for missing vehicles
        """
        # Extract step number from snap_file
        step_match = re.search(r'step_(\d+)\.json', snap_file)
        if not step_match:
            print(f"Warning: Could not extract step number from {snap_file}")
            return {}
        
        step_num = step_match.group(1)
        label_file = f"labels_{step_num}.json"
        label_path = os.path.join(labels_folder, label_file)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found at {label_path}")
            return {}
        
        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
            return {}
        
        # Load 99th percentile from the same statistics file used by dataset creator
        try:
            import pandas as pd
            label_summary_path = os.path.join(self.stats_dir, "labels_feature_summary.csv")
            label_summary_df = pd.read_csv(label_summary_path)
            ttt_row = label_summary_df[label_summary_df['feature'] == 'total_travel_time_seconds']
            if ttt_row.empty or '99%' not in ttt_row:
                print(f"Warning: Could not find 99th percentile for total_travel_time_seconds in {label_summary_path}")
                print("Falling back to calculating from current labels file...")
                # Fallback to calculating from current labels
                all_durations = []
                for label in label_data:
                    if 'total_travel_time_seconds' in label:
                        all_durations.append(label['total_travel_time_seconds'])
                if not all_durations:
                    print(f"Warning: No travel durations found in {label_path}")
                    return {}
                all_durations.sort()
                percentile_99 = all_durations[int(0.99 * len(all_durations))]
            else:
                percentile_99 = float(ttt_row['99%'].values[0])
                print(f"Using 99th percentile from statistics file: {percentile_99}")
        except Exception as e:
            print(f"Error loading statistics file: {e}")
            print("Falling back to calculating from current labels file...")
            # Fallback to calculating from current labels
            all_durations = []
            for label in label_data:
                if 'total_travel_time_seconds' in label:
                    all_durations.append(label['total_travel_time_seconds'])
            if not all_durations:
                print(f"Warning: No travel durations found in {label_path}")
                return {}
            all_durations.sort()
            percentile_99 = all_durations[int(0.99 * len(all_durations))]
        
        print(f"\n=== Missing Vehicle Duration Analysis ===")
        print(f"Label file: {label_path}")
        print(f"99th percentile of travel duration: {percentile_99:.2f} seconds")
        print(f"Valid range: 180.0 - {percentile_99:.2f} seconds")
        print(f"Missing vehicles: {len(missing_vehicle_ids)}")
        
        results = {
            'label_file': label_path,
            'percentile_99': percentile_99,
            'valid_min': 180.0,
            'valid_max': percentile_99,
            'missing_vehicles': {}
        }
        
        # Counters for summary
        filtered_too_short = 0
        filtered_too_long = 0
        unexpected_filtered = 0
        no_label_found = 0
        no_duration_found = 0
        
        # Check each missing vehicle
        for vid in missing_vehicle_ids:
            # Find the label for this vehicle
            vehicle_label = None
            for label in label_data:  # label_data is a direct list
                if label.get('vehicle_id') == vid:
                    vehicle_label = label
                    break
            
            if vehicle_label is None:
                print(f"  ❌ {vid}: No label found")
                results['missing_vehicles'][vid] = {'status': 'no_label_found'}
                no_label_found += 1
                continue
            
            duration = vehicle_label.get('total_travel_time_seconds')
            if duration is None:
                print(f"  ❌ {vid}: No total_travel_time_seconds in label")
                results['missing_vehicles'][vid] = {'status': 'no_duration_found'}
                no_duration_found += 1
                continue
            
            # Check if duration is outside valid range
            if duration < 180.0:
                print(f"  ✅ {vid}: Duration {duration:.0f}s < 180s (FILTERED - too short)")
                results['missing_vehicles'][vid] = {
                    'status': 'filtered_too_short',
                    'duration': duration,
                    'reason': f'Duration {duration:.0f}s < 180s'
                }
                filtered_too_short += 1
            elif duration > percentile_99:
                print(f"  ✅ {vid}: Duration {duration:.0f}s > {percentile_99:.0f}s (FILTERED - above 99th percentile)")
                results['missing_vehicles'][vid] = {
                    'status': 'filtered_too_long',
                    'duration': duration,
                    'reason': f'Duration {duration:.0f}s > {percentile_99:.0f}s'
                }
                filtered_too_long += 1
            else:
                print(f"  ❌ {vid}: Duration {duration:.0f}s is in valid range (UNEXPECTED - should not be filtered)")
                results['missing_vehicles'][vid] = {
                    'status': 'unexpected_filtered',
                    'duration': duration,
                    'reason': f'Duration {duration:.0f}s is in valid range'
                }
                unexpected_filtered += 1
        
        # Print summary
        print(f"\n=== Duration Analysis Summary ===")
        print(f"✅ Correctly filtered (too short): {filtered_too_short}")
        print(f"✅ Correctly filtered (too long): {filtered_too_long}")
        print(f"❌ Unexpectedly missing: {unexpected_filtered}")
        print(f"❌ No label found: {no_label_found}")
        print(f"❌ No duration found: {no_duration_found}")
        print(f"Total: {filtered_too_short + filtered_too_long + unexpected_filtered + no_label_found + no_duration_found}")
        
        return results

    def compare_restoration(self, original_json_path, restored_json, verbose=True, labels_folder=None):
        """
        Compare the restored JSON with the original snapshot to check restoration completeness.
        
        Args:
            original_json_path (str): Path to the original JSON snapshot file
            restored_json (dict): The restored JSON from restore_pt_to_json
            verbose (bool): Whether to print detailed comparison results
            labels_folder (str, optional): Path to labels folder for duration checking
            
        Returns:
            dict: Comparison results with statistics and discrepancies
        """
        try:
            with open(original_json_path, 'r') as f:
                original_json = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Original JSON file not found at {original_json_path}")
            return {"error": "Original file not found"}
        
        comparison_results = {
            "original_file": original_json_path,
            "total_nodes_original": len(original_json.get('nodes', [])),
            "total_nodes_restored": len(restored_json.get('nodes', [])),
            "total_edges_original": len(original_json.get('edges', [])),
            "total_edges_restored": len(restored_json.get('edges', [])),
            "node_discrepancies": [],
            "edge_discrepancies": [],
            "missing_fields": [],
            "extra_fields": []
        }
        
        # Compare node counts
        if comparison_results["total_nodes_original"] != comparison_results["total_nodes_restored"]:
            comparison_results["node_discrepancies"].append(
                f"Node count mismatch: {comparison_results['total_nodes_original']} vs {comparison_results['total_nodes_restored']}"
            )
        
        # Compare edge counts
        if comparison_results["total_edges_original"] != comparison_results["total_edges_restored"]:
            comparison_results["edge_discrepancies"].append(
                f"Edge count mismatch: {comparison_results['total_edges_original']} vs {comparison_results['total_edges_restored']}"
            )
        
        # Compare individual nodes
        original_nodes = original_json.get('nodes', [])
        restored_nodes = restored_json.get('nodes', [])
        
        # Create lookup dictionaries for easier comparison
        original_vehicles = {node['id']: node for node in original_nodes if 'vehicle_type' in node}
        original_junctions = {node['id']: node for node in original_nodes if 'j_type' in node}
        restored_vehicles = {node['id']: node for node in restored_nodes if 'vehicle_type' in node}
        restored_junctions = {node['id']: node for node in restored_nodes if 'j_type' in node}
        
        # Track missing vehicles for duration analysis
        missing_vehicle_ids = []
        
        # Compare vehicles
        vehicle_fields_to_compare = [
            'vehicle_type', 'speed', 'acceleration', 'route_length', 
            'current_x', 'current_y', 'destination_x', 'destination_y',
            'current_zone', 'route', 'current_edge'
        ]
        
        for vid in original_vehicles:
            if vid not in restored_vehicles:
                comparison_results["node_discrepancies"].append(f"Vehicle {vid} missing in restored data")
                missing_vehicle_ids.append(vid)
                continue
                
            orig_veh = original_vehicles[vid]
            rest_veh = restored_vehicles[vid]
            
            for field in vehicle_fields_to_compare:
                if field in orig_veh and field in rest_veh:
                    orig_val = orig_veh[field]
                    rest_val = rest_veh[field]
                    
                    # Handle different data types and precision
                    if isinstance(orig_val, (int, float)) and isinstance(rest_val, (int, float)):
                        if abs(orig_val - rest_val) > 1e-3:  # Allow larger floating point differences for coordinates
                            comparison_results["node_discrepancies"].append(
                                f"Vehicle {vid} {field}: {orig_val} vs {rest_val}"
                            )
                    elif isinstance(orig_val, str) and isinstance(rest_val, (int, float)):
                        # Handle string vs numeric comparison (e.g., "3" vs 3)
                        try:
                            orig_numeric = float(orig_val)
                            if abs(orig_numeric - rest_val) > 1e-3:
                                comparison_results["node_discrepancies"].append(
                                    f"Vehicle {vid} {field}: {orig_val} vs {rest_val}"
                                )
                        except ValueError:
                            if orig_val != str(rest_val):
                                comparison_results["node_discrepancies"].append(
                                    f"Vehicle {vid} {field}: {orig_val} vs {rest_val}"
                                )
                    elif isinstance(rest_val, str) and isinstance(orig_val, (int, float)):
                        # Handle numeric vs string comparison (e.g., 3 vs "3")
                        try:
                            rest_numeric = float(rest_val)
                            if abs(orig_val - rest_numeric) > 1e-3:
                                comparison_results["node_discrepancies"].append(
                                    f"Vehicle {vid} {field}: {orig_val} vs {rest_val}"
                                )
                        except ValueError:
                            if str(orig_val) != rest_val:
                                comparison_results["node_discrepancies"].append(
                                    f"Vehicle {vid} {field}: {orig_val} vs {rest_val}"
                                )
                    elif orig_val != rest_val:
                        comparison_results["node_discrepancies"].append(
                            f"Vehicle {vid} {field}: {orig_val} vs {rest_val}"
                        )
                elif field in orig_veh and field not in rest_veh:
                    comparison_results["missing_fields"].append(f"Vehicle {vid} missing {field}")
                elif field not in orig_veh and field in rest_veh:
                    comparison_results["extra_fields"].append(f"Vehicle {vid} extra {field}")
        
        # Check missing vehicle durations if labels folder is provided
        if labels_folder and missing_vehicle_ids:
            snap_file = os.path.basename(original_json_path)
            duration_results = self.check_missing_vehicle_durations(missing_vehicle_ids, labels_folder, snap_file)
            comparison_results["duration_analysis"] = duration_results
        
        # Compare junctions
        junction_fields_to_compare = ['x', 'y', 'zone', 'type']
        
        for jid in original_junctions:
            if jid not in restored_junctions:
                comparison_results["node_discrepancies"].append(f"Junction {jid} missing in restored data")
                continue
                
            orig_junc = original_junctions[jid]
            rest_junc = restored_junctions[jid]
            
            for field in junction_fields_to_compare:
                if field in orig_junc and field in rest_junc:
                    orig_val = orig_junc[field]
                    rest_val = rest_junc[field]
                    
                    # Handle different data types and precision
                    if isinstance(orig_val, (int, float)) and isinstance(rest_val, (int, float)):
                        if abs(orig_val - rest_val) > 1e-3:
                            comparison_results["node_discrepancies"].append(
                                f"Junction {jid} {field}: {orig_val} vs {rest_val}"
                            )
                    elif isinstance(orig_val, str) and isinstance(rest_val, (int, float)):
                        # Handle string vs numeric comparison (e.g., "3" vs 3)
                        try:
                            orig_numeric = float(orig_val)
                            if abs(orig_numeric - rest_val) > 1e-3:
                                comparison_results["node_discrepancies"].append(
                                    f"Junction {jid} {field}: {orig_val} vs {rest_val}"
                                )
                        except ValueError:
                            if orig_val != str(rest_val):
                                comparison_results["node_discrepancies"].append(
                                    f"Junction {jid} {field}: {orig_val} vs {rest_val}"
                                )
                    elif isinstance(rest_val, str) and isinstance(orig_val, (int, float)):
                        # Handle numeric vs string comparison (e.g., 3 vs "3")
                        try:
                            rest_numeric = float(rest_val)
                            if abs(orig_val - rest_numeric) > 1e-3:
                                comparison_results["node_discrepancies"].append(
                                    f"Junction {jid} {field}: {orig_val} vs {rest_val}"
                                )
                        except ValueError:
                            if str(orig_val) != rest_val:
                                comparison_results["node_discrepancies"].append(
                                    f"Junction {jid} {field}: {orig_val} vs {rest_val}"
                                )
                    elif orig_val != rest_val:
                        comparison_results["node_discrepancies"].append(
                            f"Junction {jid} {field}: {orig_val} vs {rest_val}"
                        )
        
        # Compare edges
        original_edges = {edge['id']: edge for edge in original_json.get('edges', [])}
        restored_edges = {edge['id']: edge for edge in restored_json.get('edges', [])}
        
        edge_fields_to_compare = ['num_lanes', 'avg_speed', 'length', 'edge_demand', 'edge_occupancy']
        
        for eid in original_edges:
            if eid not in restored_edges:
                comparison_results["edge_discrepancies"].append(f"Edge {eid} missing in restored data")
                continue
                
            orig_edge = original_edges[eid]
            rest_edge = restored_edges[eid]
            
            for field in edge_fields_to_compare:
                if field in orig_edge and field in rest_edge:
                    orig_val = orig_edge[field]
                    rest_val = rest_edge[field]
                    
                    # Handle different data types and precision
                    if isinstance(orig_val, (int, float)) and isinstance(rest_val, (int, float)):
                        if abs(orig_val - rest_val) > 1e-3:
                            comparison_results["edge_discrepancies"].append(
                                f"Edge {eid} {field}: {orig_val} vs {rest_val}"
                            )
                    elif isinstance(orig_val, str) and isinstance(rest_val, (int, float)):
                        # Handle string vs numeric comparison (e.g., "3" vs 3)
                        try:
                            orig_numeric = float(orig_val)
                            if abs(orig_numeric - rest_val) > 1e-3:
                                comparison_results["edge_discrepancies"].append(
                                    f"Edge {eid} {field}: {orig_val} vs {rest_val}"
                                )
                        except ValueError:
                            if orig_val != str(rest_val):
                                comparison_results["edge_discrepancies"].append(
                                    f"Edge {eid} {field}: {orig_val} vs {rest_val}"
                                )
                    elif isinstance(rest_val, str) and isinstance(orig_val, (int, float)):
                        # Handle numeric vs string comparison (e.g., 3 vs "3")
                        try:
                            rest_numeric = float(rest_val)
                            if abs(orig_val - rest_numeric) > 1e-3:
                                comparison_results["edge_discrepancies"].append(
                                    f"Edge {eid} {field}: {orig_val} vs {rest_val}"
                                )
                        except ValueError:
                            if str(orig_val) != rest_val:
                                comparison_results["edge_discrepancies"].append(
                                    f"Edge {eid} {field}: {orig_val} vs {rest_val}"
                                )
                    elif orig_val != rest_val:
                        comparison_results["edge_discrepancies"].append(
                            f"Edge {eid} {field}: {orig_val} vs {rest_val}"
                        )
        
        # Check missing vehicle durations if labels folder is provided
        if labels_folder and missing_vehicle_ids:
            print(f"\n=== Checking Missing Vehicle Durations ===")
            duration_results = self.check_missing_vehicle_durations(missing_vehicle_ids, labels_folder, original_json_path)
            comparison_results["duration_analysis"] = duration_results
        
        # Print summary if verbose
        if verbose:
            print(f"\n=== Restoration Comparison Results ===")
            print(f"Original file: {original_json_path}")
            print(f"Nodes: {comparison_results['total_nodes_original']} original vs {comparison_results['total_nodes_restored']} restored")
            print(f"Edges: {comparison_results['total_edges_original']} original vs {comparison_results['total_edges_restored']} restored")
            
            if comparison_results["node_discrepancies"]:
                print(f"\nNode discrepancies ({len(comparison_results['node_discrepancies'])}):")
                for disc in comparison_results["node_discrepancies"][:10]:  # Show first 10
                    print(f"  - {disc}")
                if len(comparison_results["node_discrepancies"]) > 10:
                    print(f"  ... and {len(comparison_results['node_discrepancies']) - 10} more")
            
            if comparison_results["edge_discrepancies"]:
                print(f"\nEdge discrepancies ({len(comparison_results['edge_discrepancies'])}):")
                for disc in comparison_results["edge_discrepancies"][:10]:  # Show first 10
                    print(f"  - {disc}")
                if len(comparison_results["edge_discrepancies"]) > 10:
                    print(f"  ... and {len(comparison_results['edge_discrepancies']) - 10} more")
            
            if comparison_results["missing_fields"]:
                print(f"\nMissing fields ({len(comparison_results['missing_fields'])}):")
                for field in comparison_results["missing_fields"][:5]:  # Show first 5
                    print(f"  - {field}")
                if len(comparison_results["missing_fields"]) > 5:
                    print(f"  ... and {len(comparison_results['missing_fields']) - 5} more")
            
            if comparison_results["extra_fields"]:
                print(f"\nExtra fields ({len(comparison_results['extra_fields'])}):")
                for field in comparison_results["extra_fields"][:5]:  # Show first 5
                    print(f"  - {field}")
                if len(comparison_results["extra_fields"]) > 5:
                    print(f"  ... and {len(comparison_results['extra_fields']) - 5} more")
            
            total_issues = (len(comparison_results["node_discrepancies"]) + 
                          len(comparison_results["edge_discrepancies"]) + 
                          len(comparison_results["missing_fields"]) + 
                          len(comparison_results["extra_fields"]))
            
            if total_issues == 0:
                print(f"\n✅ Restoration appears to be complete and accurate!")
            else:
                print(f"\n⚠️  Found {total_issues} issues in restoration")
        
        return comparison_results

    def restore_pt_to_json(self, pt_path, output_json_path=None):
        """
        Main method to restore a .pt file to JSON.
        Args:
            pt_path (str): Path to the .pt file.
            output_json_path (str, optional): If provided, save the JSON to this path.
        Returns:
            dict: Restored JSON structure.
        """
        data = torch.load(pt_path)
        # Restore junctions and vehicles
        restored_junctions = self.restore_junctions_from_pt(data)
        restored_vehicles = self.restore_vehicles_from_pt(data)
        # Restore edges
        restored_edges = self.restore_edges_from_pt(data)
        # Combine nodes
        nodes = restored_vehicles + restored_junctions
        # Restore step if available
        step = getattr(data, 'step', None)
        restored_json = {'nodes': nodes, 'edges': restored_edges}
        if step is not None:
            restored_json['step'] = int(step)
        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(to_python_type(restored_json), f, indent=2)
        return restored_json

def main():
    parser = argparse.ArgumentParser(description="Restore a PyG .pt file to a human-readable JSON snapshot.")
    parser.add_argument('--pt_file', type=str, required=True, help='Path to the .pt file to restore.')
    parser.add_argument('--stats_dir', type=str, default='/home/guy/Projects/Traffic/Traffic-DSTG-Gen/eda_exports', help='Directory with normalization statistics (default: eda_exports)')
    parser.add_argument('--mapping_dir', type=str, default='/home/guy/Projects/Traffic/Traffic-DSTG-Gen/eda_exports/mappings', help='Directory with mapping files (default: eda_exports/mappings)')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save the restored JSON (optional)')
    parser.add_argument('--compare_with', type=str, default=None, help='Path to original JSON file to compare restoration accuracy')
    parser.add_argument('--labels_folder', type=str, default=None, help='Path to labels folder for duration analysis of missing vehicles')
    args = parser.parse_args()

    restorer = DatasetRestorer(stats_dir=args.stats_dir, mapping_dir=args.mapping_dir)
    restored = restorer.restore_pt_to_json(args.pt_file, args.output_json)
    
    # Compare with original if provided
    if args.compare_with:
        comparison_results = restorer.compare_restoration(args.compare_with, restored, labels_folder=args.labels_folder)
        if comparison_results.get("error"):
            print(f"Comparison failed: {comparison_results['error']}")
    
    if args.output_json is None:
        print(json.dumps(to_python_type(restored), indent=2))
    else:
        print(f"Restored JSON saved to {args.output_json}")

if __name__ == '__main__':
    main() 