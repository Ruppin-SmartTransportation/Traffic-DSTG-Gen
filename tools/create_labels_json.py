import os
import json
from tqdm import tqdm
import argparse
import sys

def create_snapshot_labels(
    snapshots_folder,
    gt_labels_path,
    output_labels_folder
):
    # Load ground truth journey log
    with open(gt_labels_path, "r") as f:
        gt_list = json.load(f)

    # Map: vehicle_id -> list of all its trips (GT entries)
    gt_map = {}
    for entry in gt_list:
        vid = entry["vehicle_id"]
        gt_map.setdefault(vid, []).append(entry)

    os.makedirs(output_labels_folder, exist_ok=True)

    snapshot_files = [
        f for f in os.listdir(snapshots_folder)
        if f.endswith(".json") and "labels" not in f
    ]
    
    error_files = []
    processed_count = 0
    
    for snap_file in tqdm(snapshot_files, desc="Creating GT labels per snapshot"):
        snap_path = os.path.join(snapshots_folder, snap_file)
        
        try:
            with open(snap_path, "r") as f:
                snap = json.load(f)
        except (OSError, IOError, json.JSONDecodeError) as e:
            print(f"\nError reading file {snap_file}: {e}")
            error_files.append(snap_file)
            continue
        except Exception as e:
            print(f"\nUnexpected error reading file {snap_file}: {e}")
            error_files.append(snap_file)
            continue
            
        # Get snapshot time in seconds from the JSON content
        snap_time = snap.get("step")

        labels = []
        for node in snap.get("nodes", []):
            if node.get("node_type") == 1:
                vid = node["id"]
                gt_trips = gt_map.get(vid, [])
                
                # Find the trip for this snapshot
                matching_gt = None
                
                # First, try to find an active trip (vehicle is currently traveling)
                for trip in gt_trips:
                    if trip["origin_time_sec"] <= snap_time <= trip["destination_time_sec"]:
                        matching_gt = trip
                        break
                
                # If no active trip found, find the most relevant trip based on timing
                if not matching_gt and gt_trips:
                    # Sort trips by start time to find the most logical sequence
                    sorted_trips = sorted(gt_trips, key=lambda x: x["origin_time_sec"])
                    
                    # Find the most recent completed trip
                    completed_trips = [trip for trip in sorted_trips if trip["destination_time_sec"] < snap_time]
                    
                    # Find the next upcoming trip
                    upcoming_trips = [trip for trip in sorted_trips if trip["origin_time_sec"] > snap_time]
                    
                    if completed_trips:
                        # Use the most recent completed trip
                        matching_gt = completed_trips[-1]
                    elif upcoming_trips:
                        # Use the next upcoming trip
                        matching_gt = upcoming_trips[0]
                    else:
                        # This shouldn't happen if we have trips, but fallback to first trip
                        matching_gt = sorted_trips[0]
                
                if not matching_gt:
                    continue  # Skip if no ground truth found at all
                
                # Calculate ETA based on trip status
                if matching_gt["origin_time_sec"] <= snap_time <= matching_gt["destination_time_sec"]:
                    # Vehicle is actively traveling
                    eta = max(matching_gt["destination_time_sec"] - snap_time, 0)
                elif snap_time < matching_gt["origin_time_sec"]:
                    # Vehicle hasn't started yet - ETA is time until start + full trip duration
                    eta = (matching_gt["origin_time_sec"] - snap_time) + matching_gt["total_travel_time_seconds"]
                else:
                    # Vehicle has completed the trip - ETA is 0
                    eta = 0
                
                labels.append({
                    "vehicle_id": vid,
                    "origin_time_sec": matching_gt["origin_time_sec"],
                    "destination_time_sec": matching_gt["destination_time_sec"],
                    "total_travel_time_seconds": matching_gt["total_travel_time_seconds"],
                    "eta": eta
                })
        
        try:
            out_name = snap_file.replace("step_", "labels_")
            out_path = os.path.join(output_labels_folder, out_name)
            with open(out_path, "w") as f:
                json.dump(labels, f, indent=2)
            processed_count += 1
        except (OSError, IOError) as e:
            print(f"\nError writing file {out_name}: {e}")
            error_files.append(snap_file)
            continue
    
    print(f"\nDone! Processed {processed_count} files successfully.")
    if error_files:
        print(f"Failed to process {len(error_files)} files:")
        for error_file in error_files[:10]:  # Show first 10 errors
            print(f"  - {error_file}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more files")
        print(f"Snapshot label files written to {output_labels_folder}")
    else:
        print(f"All files processed successfully! Snapshot label files written to {output_labels_folder}")

def main():
    parser = argparse.ArgumentParser(description="Generate per-snapshot ETA ground truth label files")
    parser.add_argument(
        "--snapshots_folder",
        type=str,
        help="Path to folder with snapshot JSON files"
    )
    parser.add_argument(
        "--gt_labels_path",
        type=str,
        help="Path to ground truth labels.json file"
    )
    parser.add_argument(
        "--output_labels_folder",
        type=str,
        help="Folder where per-snapshot label files will be written"
    )
    args = parser.parse_args()

    create_snapshot_labels(
        args.snapshots_folder,
        args.gt_labels_path,
        args.output_labels_folder
    )

if __name__ == "__main__":
    main()
