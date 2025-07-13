import os
import json
from tqdm import tqdm
import argparse

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
    for snap_file in tqdm(snapshot_files, desc="Creating GT labels per snapshot"):
        snap_path = os.path.join(snapshots_folder, snap_file)
        with open(snap_path, "r") as f:
            snap = json.load(f)
        # Get snapshot time in seconds from the JSON content
        snap_time = snap.get("step")

        labels = []
        for node in snap.get("nodes", []):
            if node.get("node_type") == 1:
                vid = node["id"]
                gt_trips = gt_map.get(vid, [])
                # Find the trip for this snapshot
                matching_gt = None
                for trip in gt_trips:
                    # FIXED: Remove arbitrary +1 offset from time range check
                    if trip["origin_time_sec"] <= snap_time <= trip["destination_time_sec"]:
                        matching_gt = trip
                        break
                if not matching_gt:
                    continue  # Skip if not found in GT for this time
                # FIXED: Remove arbitrary +1 offset from ETA calculation
                eta = max(matching_gt["destination_time_sec"] - snap_time, 0)
                labels.append({
                    "vehicle_id": vid,
                    "origin_time_sec": matching_gt["origin_time_sec"],
                    "destination_time_sec": matching_gt["destination_time_sec"],
                    "total_travel_time_seconds": matching_gt["total_travel_time_seconds"],
                    "eta": eta
                })
        out_name = snap_file.replace("step_", "labels_")
        out_path = os.path.join(output_labels_folder, out_name)
        with open(out_path, "w") as f:
            json.dump(labels, f, indent=2)
    print(f"Done! Snapshot label files written to {output_labels_folder}")

def main():
    parser = argparse.ArgumentParser(description="Generate per-snapshot ETA ground truth label files")
    parser.add_argument(
        "--snapshots_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data",
        help="Path to folder with snapshot JSON files"
    )
    parser.add_argument(
        "--gt_labels_path",
        type=str,
        default="/media/guy/StorageVolume/traffic_data/labels.json",
        help="Path to ground truth labels.json file"
    )
    parser.add_argument(
        "--output_labels_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data/labels",
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
