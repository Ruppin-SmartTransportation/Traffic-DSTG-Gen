#!/usr/bin/env python3
"""
Validate label files against snapshot files
This script checks the first 10 snapshot-label pairs to ensure:
1. All vehicles in snapshots are in corresponding labels
2. ETA calculations are correct
3. All required fields are present
4. No missing data
"""

import os
import json
import re
from collections import defaultdict

def extract_timestamp_from_filename(filename):
    """Extract timestamp from filename like 'step_000060.json' -> 60"""
    match = re.search(r'step_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    return None

def validate_snapshot_label_pair(snapshot_path, label_path, snapshots_folder, labels_folder):
    """Validate a single snapshot-label pair"""
    
    # Extract timestamp from filename
    snapshot_filename = os.path.basename(snapshot_path)
    timestamp = extract_timestamp_from_filename(snapshot_filename)
    
    print(f"\n🔍 Validating: {snapshot_filename} -> {os.path.basename(label_path)}")
    print(f"   Timestamp: {timestamp} seconds")
    
    # Load snapshot
    try:
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
    except Exception as e:
        print(f"   ❌ Failed to load snapshot: {e}")
        return False
    
    # Load label
    try:
        with open(label_path, 'r') as f:
            labels = json.load(f)
    except Exception as e:
        print(f"   ❌ Failed to load label: {e}")
        return False
    
    # Load ground truth to check which vehicles should be active
    try:
        with open("/media/guy/StorageVolume/traffic_data/labels.json", 'r') as f:
            gt_list = json.load(f)
    except Exception as e:
        print(f"   ❌ Failed to load ground truth: {e}")
        return False
    
    # Create ground truth map
    gt_map = {}
    for entry in gt_list:
        vid = entry["vehicle_id"]
        gt_map.setdefault(vid, []).append(entry)
    
    # Get vehicles from snapshot
    snapshot_vehicles = []
    for node in snapshot.get("nodes", []):
        if node.get("node_type") == 1:  # Vehicle node
            snapshot_vehicles.append(node["id"])
    
    # Get vehicles from labels
    label_vehicles = [label["vehicle_id"] for label in labels]
    
    print(f"   🚗 Vehicles in snapshot: {len(snapshot_vehicles)}")
    print(f"   🏷️  Vehicles in label: {len(label_vehicles)}")
    
    # Check which vehicles should be active at this timestamp
    expected_active_vehicles = []
    for vid in snapshot_vehicles:
        gt_trips = gt_map.get(vid, [])
        for trip in gt_trips:
            if trip["origin_time_sec"] <= timestamp <= trip["destination_time_sec"]:
                expected_active_vehicles.append(vid)
                break
    
    print(f"   ✅ Expected active vehicles: {len(expected_active_vehicles)}")
    
    # Check if all expected active vehicles are in labels
    missing_vehicles = set(expected_active_vehicles) - set(label_vehicles)
    extra_vehicles = set(label_vehicles) - set(expected_active_vehicles)
    
    if missing_vehicles:
        print(f"   ❌ Missing vehicles in label: {list(missing_vehicles)[:5]}{'...' if len(missing_vehicles) > 5 else ''}")
        return False
    
    if extra_vehicles:
        print(f"   ⚠️  Extra vehicles in label: {list(extra_vehicles)[:5]}{'...' if len(extra_vehicles) > 5 else ''}")
        return False
    
    print(f"   ✅ Vehicle matching: Perfect")
    
    # Validate each label entry
    validation_errors = []
    
    for i, label in enumerate(labels):
        # Check required fields
        required_fields = ["vehicle_id", "origin_time_sec", "destination_time_sec", "total_travel_time_seconds", "eta"]
        for field in required_fields:
            if field not in label:
                validation_errors.append(f"Label {i}: Missing field '{field}'")
                continue
            if label[field] is None:
                validation_errors.append(f"Label {i}: Field '{field}' is None")
                continue
        
        # Validate ETA calculation
        if "eta" in label and "destination_time_sec" in label:
            expected_eta = max(label["destination_time_sec"] - timestamp, 0)
            if label["eta"] != expected_eta:
                validation_errors.append(f"Label {i} ({label['vehicle_id']}): ETA mismatch - expected {expected_eta}, got {label['eta']}")
        
        # Validate total travel time
        if "origin_time_sec" in label and "destination_time_sec" in label and "total_travel_time_seconds" in label:
            expected_total = label["destination_time_sec"] - label["origin_time_sec"]
            if label["total_travel_time_seconds"] != expected_total:
                validation_errors.append(f"Label {i} ({label['vehicle_id']}): Total travel time mismatch - expected {expected_total}, got {label['total_travel_time_seconds']}")
        
        # Validate time range
        if "origin_time_sec" in label and "destination_time_sec" in label:
            if label["origin_time_sec"] > label["destination_time_sec"]:
                validation_errors.append(f"Label {i} ({label['vehicle_id']}): Invalid time range - origin > destination")
    
    if validation_errors:
        print(f"   ❌ Validation errors:")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"      {error}")
        if len(validation_errors) > 5:
            print(f"      ... and {len(validation_errors) - 5} more")
        return False
    
    print(f"   ✅ All labels valid")
    
    # Show sample label structure
    if labels:
        print(f"   📄 Sample label structure:")
        sample = labels[0]
        for key, value in sample.items():
            print(f"      {key}: {value} ({type(value).__name__})")
    
    return True

def main():
    snapshots_folder = "/media/guy/StorageVolume/traffic_data"
    labels_folder = "/media/guy/StorageVolume/traffic_data/labels"
    
    print("🔍 Validating Label Files Against Snapshots")
    print("=" * 60)
    
    # Get first 10 snapshot files
    snapshot_files = []
    for f in os.listdir(snapshots_folder):
        if f.startswith("step_") and f.endswith(".json"):
            snapshot_files.append(f)
    snapshot_files.sort()
    snapshot_files = snapshot_files[:10]
    
    print(f"📁 Found {len(snapshot_files)} snapshot files to validate")
    
    validation_results = []
    
    for snapshot_file in snapshot_files:
        # Construct corresponding label filename
        label_file = snapshot_file.replace("step_", "labels_")
        
        snapshot_path = os.path.join(snapshots_folder, snapshot_file)
        label_path = os.path.join(labels_folder, label_file)
        
        # Check if label file exists
        if not os.path.exists(label_path):
            print(f"❌ Label file not found: {label_file}")
            validation_results.append(False)
            continue
        
        # Validate the pair
        is_valid = validate_snapshot_label_pair(snapshot_path, label_path, snapshots_folder, labels_folder)
        validation_results.append(is_valid)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    valid_count = sum(validation_results)
    total_count = len(validation_results)
    
    print(f"Files validated: {total_count}")
    print(f"Valid pairs: {valid_count}")
    print(f"Invalid pairs: {total_count - valid_count}")
    
    if valid_count == total_count:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ All vehicles are properly matched")
        print("✅ All ETA calculations are correct")
        print("✅ All required fields are present")
        print("✅ No missing data")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please check the errors above and fix the issues")
    
    return valid_count == total_count

if __name__ == "__main__":
    main() 