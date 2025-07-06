import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import argparse
import sys
from datetime import datetime, timedelta

SAMPLE_SIZE = 3000  # <-- Set the number of snapshot files to sample

def safe_filename(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def get_snapshot_files(snapshots_folder):
    """Return up to SAMPLE_SIZE snapshot files (excluding any 'labels' file)."""
    all_files = [
        os.path.join(snapshots_folder, f)
        for f in os.listdir(snapshots_folder)
        if f.endswith('.json') and "labels" not in f
    ]
    if len(all_files) > SAMPLE_SIZE:
        all_files = random.sample(all_files, SAMPLE_SIZE)
        print(f"Randomly sampled {SAMPLE_SIZE} snapshot files for analysis.")
    return all_files

def plot_route_distances_over_time(labels_json_path, output_folder="eda_exports"):

    start_day = int(input("Enter start day (0–28): "))
    start_hour = int(input("Enter start hour (0–24): "))
    end_day = int(input("Enter end day (exclusive, 0–28): "))
    end_hour = int(input("Enter end hour (exclusive, 0–24): "))

    start_sec = start_day * 86400 + start_hour * 3600
    end_sec = end_day * 86400 + end_hour * 3600

    times = []
    distances_km = []

    with open(labels_json_path) as f:
        labels = json.load(f)
    for entry in labels:
        t = entry.get("origin_time_sec", 0)
        if start_sec <= t < end_sec:
            times.append(datetime.utcfromtimestamp(t))
            distances_km.append(entry.get("initial_route_length", 0.0) / 1000.0)

    if not times:
        print("No entries found for the specified time window.")
        return

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, distances_km, '.', alpha=0.5)
    plt.title("Route Distance over Time")
    plt.xlabel("Time")
    plt.ylabel("Distance (km)")

    plt.subplot(1, 2, 2)
    plt.hist(distances_km, bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Route Distances")
    plt.xlabel("Distance (km)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    fname = os.path.join(output_folder, "route_distances_over_time.pdf")
    plt.savefig(fname)

def plot_travel_durations_over_time(labels_json_path, output_folder="eda_exports"):
    
    start_day = int(input("Enter start day (0–28): "))
    start_hour = int(input("Enter start hour (0–24): "))
    end_day = int(input("Enter end day (exclusive, 0–28): "))
    end_hour = int(input("Enter end hour (exclusive, 0–24): "))

    start_sec = start_day * 86400 + start_hour * 3600
    end_sec = end_day * 86400 + end_hour * 3600

    times = []
    durations_min = []
    
    with open(labels_json_path) as f:
        labels = json.load(f)

    for entry in labels:
        t = entry.get("origin_time_sec", 0)
        if start_sec <= t < end_sec:
            times.append(datetime.utcfromtimestamp(t))
            durations_min.append(entry.get("total_travel_time_seconds", 0) / 60.0)

    if not times:
        print("No entries found for the specified time window.")
        return

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, durations_min, '.', alpha=0.5)
    plt.title("Travel Duration over Time")
    plt.xlabel("Time")
    plt.ylabel("Duration (min)")

    plt.subplot(1, 2, 2)
    plt.hist(durations_min, bins=30, color='orange', edgecolor='black')
    plt.title("Histogram of Travel Durations")
    plt.xlabel("Duration (min)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
    fname = os.path.join(output_folder, "travel_durations_over_time.pdf")
    plt.savefig(fname)

def analyze_feature_distribution(
    snapshots_folder,
    feature_name,
    entity_type,
    output_options=None,
    sample_size=None,
    export_folder="./eda_exports"
):
    """
    Analyze the distribution of a specific feature for a given entity type.
    Exports plots and stats to export_folder.
    """
    os.makedirs(export_folder, exist_ok=True)
    all_files = get_snapshot_files(snapshots_folder)
    values = []
    for file in tqdm(all_files, desc=f"Scanning files for {feature_name} ({entity_type})"):
        with open(file, 'r') as f:
            data = json.load(f)
            if entity_type == "vehicle":
                for node in data.get("nodes", []):
                    if node.get("node_type") == 1:
                        v = node.get(feature_name)
                        if v is not None:
                            values.append(v)
            elif entity_type == "junction":
                for node in data.get("nodes", []):
                    if node.get("node_type") == 0:
                        v = node.get(feature_name)
                        if v is not None:
                            values.append(v)
            elif entity_type == "edge":
                for edge in data.get("edges", []):
                    v = edge.get(feature_name)
                    if v is not None:
                        values.append(v)
    if not values:
        print(f"No values found for feature '{feature_name}'.")
        return

    if sample_size and sample_size < len(values):
        values = random.sample(values, sample_size)
    values = np.array(values)
    if output_options is None:
        output_options = ["histogram", "boxplot", "stats"]

    base = f"{entity_type}_{safe_filename(feature_name)}"

    if "histogram" in output_options:
        plt.figure()
        plt.hist(values, bins=50)
        plt.title(f"Histogram of {feature_name}")
        plt.xlabel(feature_name)
        plt.ylabel("Count")
        fname = os.path.join(export_folder, f"{base}_histogram.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved histogram to {fname}")

    if "boxplot" in output_options:
        plt.figure()
        plt.boxplot(values, vert=False)
        plt.title(f"Boxplot of {feature_name}")
        plt.xlabel(feature_name)
        fname = os.path.join(export_folder, f"{base}_boxplot.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved boxplot to {fname}")

    if "stats" in output_options:
        statstr = (
            f"Statistics for {feature_name} ({entity_type}):\n"
            f"Count: {len(values)}\n"
            f"Mean: {np.mean(values):.3f}\n"
            f"Std: {np.std(values):.3f}\n"
            f"Min: {np.min(values):.3f}\n"
            f"Max: {np.max(values):.3f}\n"
            f"Skewness: {skew(values):.3f}\n"
            f"Kurtosis: {kurtosis(values):.3f}\n"
        )
        fname = os.path.join(export_folder, f"{base}_stats.txt")
        with open(fname, "w") as f:
            f.write(statstr)
        print(f"Saved stats to {fname}")

    if "skewness" in output_options:
        plt.figure()
        pd.Series(values).plot(kind='kde', title=f"KDE of {feature_name} (Skewness: {skew(values):.2f})")
        plt.xlabel(feature_name)
        fname = os.path.join(export_folder, f"{base}_kde_skewness.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved skewness plot to {fname}")

    if "normalization_preview" in output_options:
        minmax = (values - np.min(values)) / (np.max(values) - np.min(values))
        zscore = (values - np.mean(values)) / np.std(values)
        plt.figure()
        plt.hist(minmax, bins=50, alpha=0.7, label='Min-max')
        plt.hist(zscore, bins=50, alpha=0.7, label='Z-score')
        plt.title(f"Normalized distributions of {feature_name}")
        plt.legend()
        fname = os.path.join(export_folder, f"{base}_normalization_preview.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved normalization preview to {fname}")

    if "outliers" in output_options:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = values[(values < lower) | (values > upper)]
        outlierstr = f"Found {len(outliers)} outliers ({100 * len(outliers)/len(values):.2f}%) for {feature_name}.\n"
        if len(outliers) > 0:
            outlierstr += f"Outlier values (first 10): {outliers[:10]}\n"
        fname = os.path.join(export_folder, f"{base}_outliers.txt")
        with open(fname, "w") as f:
            f.write(outlierstr)
        print(f"Saved outlier report to {fname}")
        plt.figure()
        plt.scatter(range(len(values)), values, alpha=0.5, label='All')
        plt.scatter(np.where((values < lower) | (values > upper)), outliers, color='red', label='Outliers')
        plt.title(f"Outliers in {feature_name}")
        plt.xlabel("Sample Index")
        plt.ylabel(feature_name)
        plt.legend()
        fname = os.path.join(export_folder, f"{base}_outliers_plot.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved outlier plot to {fname}")

def get_available_features(snapshots_folder, entity_type):
    """
    Returns a sorted list of features for the specified entity type
    """
    all_files = get_snapshot_files(snapshots_folder)
    if not all_files:
        return []
    with open(all_files[0], 'r') as f:
        data = json.load(f)
        features = set()
        if entity_type == "vehicle":
            for node in data.get("nodes", []):
                if node.get("node_type") == 1:
                    features.update(node.keys())
        elif entity_type == "junction":
            for node in data.get("nodes", []):
                if node.get("node_type") == 0:
                    features.update(node.keys())
        elif entity_type == "edge":
            for edge in data.get("edges", []):
                features.update(edge.keys())
    return sorted(features)

def summarize_labels(labels_folder, export_folder="./eda_exports"):
    print("\nProcessing label files...")
    os.makedirs(export_folder, exist_ok=True)
    all_files = [
        os.path.join(labels_folder, f)
        for f in os.listdir(labels_folder)
        if f.startswith("labels_") and f.endswith(".json")
    ]

    feature_values = dict()
    for file in tqdm(all_files, desc="Scanning label files"):
        with open(file, 'r') as f:
            label_list = json.load(f)
            for entry in label_list:
                for k, v in entry.items():
                    feature_values.setdefault(k, []).append(v)

    rows = []
    for feat, vals in feature_values.items():
        if all(isinstance(v, (list, dict)) or v is None for v in vals):
            print(f"Skipping complex feature '{feat}' (all values are lists/dicts).")
            continue
        vals_clean = [v if v not in [None, "", "None"] else np.nan for v in vals]
        try:
            vals_num = pd.to_numeric(pd.Series(vals_clean), errors='coerce')
            is_numeric = not np.all(np.isnan(vals_num))
        except Exception:
            is_numeric = False
        if is_numeric:
            vals_for_stats = vals_num.dropna()
            summary = {
                "feature": feat,
                "type": "numeric",
                "count": len(vals_clean),
                "mean": np.mean(vals_for_stats),
                "std": np.std(vals_for_stats),
                "min": np.min(vals_for_stats),
                "25%": np.percentile(vals_for_stats, 25),
                "median": np.median(vals_for_stats),
                "75%": np.percentile(vals_for_stats, 75),
                "max": np.max(vals_for_stats),
                "skewness": skew(vals_for_stats, nan_policy="omit"),
                "kurtosis": kurtosis(vals_for_stats, nan_policy="omit"),
                "num_missing": sum(pd.isna(vals_clean)),
                "num_unique": pd.Series(vals_clean).nunique()
            }
        else:
            summary = {
                "feature": feat,
                "type": "categorical",
                "count": len(vals_clean),
                "num_missing": sum(pd.isna(vals_clean)),
                "num_unique": len(set(vals_clean))
            }
        rows.append(summary)

    df = pd.DataFrame(rows)
    outpath = os.path.join(export_folder, "labels_feature_summary.csv")
    df.to_csv(outpath, index=False)
    print(f"Saved label summary to {outpath}")


def summarize_features_for_preprocessing(
    snapshots_folder,
    export_folder="./eda_exports"
):
    from scipy.stats import skew, kurtosis
    import os
    import json
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    def _safe_filename(s):
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

    def _summarize_features(values, feature_name=None):
        s = pd.Series(values)
        summary = {
            "count": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "25%": s.quantile(0.25),
            "median": s.median(),
            "75%": s.quantile(0.75),
            "max": s.max(),
            "skewness": skew(s, nan_policy="omit"),
            "kurtosis": kurtosis(s, nan_policy="omit"),
            "num_missing": s.isnull().sum(),
            "num_unique": s.nunique(),
            "normalization": "z-score" if (
                feature_name and (
                    feature_name.endswith("_log") or  # <-- only normalize log versions
                    feature_name in ["speed", "length", "avg_speed"]
                )
            ) else None
        }
        if summary["num_unique"] < 20:
            summary["value_counts"] = s.value_counts().to_dict()
        return summary

    os.makedirs(export_folder, exist_ok=True)
    all_files = get_snapshot_files(snapshots_folder)

    entity_types = {
        "vehicle": lambda node: node.get("node_type") == 1,
        "junction": lambda node: node.get("node_type") == 0,
        "edge": None,  # For "edges" array
    }

    for entity, filter_fn in entity_types.items():
        print(f"\nProcessing {entity} features...")
        feature_values = dict()
        for file in tqdm(all_files, desc=f"Scanning {entity} files"):
            with open(file, 'r') as f:
                data = json.load(f)
                if entity == "edge":
                    items = data.get("edges", [])
                else:
                    items = [n for n in data.get("nodes", []) if filter_fn(n)]
                for item in items:
                    for k, v in item.items():
                        # Special handling for vehicles_on_road → count only
                        if k == "vehicles_on_road":
                            count = len(v) if isinstance(v, list) else np.nan
                            feature_values.setdefault("vehicles_on_road_count", []).append(count)
                            feature_values.setdefault("vehicles_on_road_count_log", []).append(np.log1p(count))
                            continue
                        # Default
                        feature_values.setdefault(k, []).append(v)

        rows = []
        for feat, vals in feature_values.items():
            # Skip complex structures except vehicles_on_road_count (which is now numeric)
            if all(isinstance(v, (list, dict)) or v is None for v in vals):
                print(f"Skipping complex feature '{feat}' (all values are lists/dicts).")
                continue

            vals_clean = [v if v not in [None, "", "None"] else np.nan for v in vals]
            try:
                vals_num = pd.to_numeric(pd.Series(vals_clean), errors='coerce')
                is_numeric = not np.all(np.isnan(vals_num))
            except Exception:
                is_numeric = False
                
            if is_numeric:
                vals_for_stats = vals_num.dropna()
                summary = _summarize_features(vals_for_stats, feature_name=feat)
            else:
                scalars = [v for v in vals_clean if not isinstance(v, (list, dict))]
                summary = {
                    "count": len(vals_clean),
                    "num_missing": sum(pd.isna(vals_clean)),
                    "num_unique": len(set(scalars)),
                    "value_counts": pd.Series(scalars).value_counts().to_dict() if len(set(scalars)) < 20 else None
                }
            row = {
                "feature": feat,
                "type": "numeric" if is_numeric else "categorical"
            }
            row.update(summary)
            rows.append(row)

        df = pd.DataFrame(rows)
        outpath = os.path.join(export_folder, f"{entity}_feature_summary.csv")
        df.to_csv(outpath, index=False)
        print(f"Saved {entity} summary to {outpath}")

def print_menu(options, prompt="Select an option:"):
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        choice = input(f"{prompt} (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        else:
            print("Invalid choice. Try again.")

def analyze_edge_route_counts(snapshots_folder, export_folder="./eda_exports", plot_histogram=True):

    os.makedirs(export_folder, exist_ok=True)
    all_files = get_snapshot_files(snapshots_folder)
    if not all_files:
        print("No snapshot files found.")
        return

    per_snapshot_counts = []
    per_snapshot_logs = []
    edge_ids = set()

    for file in tqdm(all_files, desc="Processing snapshots"):
        with open(file, 'r') as f:
            snapshot = json.load(f)

        edges = snapshot.get("edges", [])
        current_edge_ids = {edge["id"] for edge in edges}
        edge_ids |= current_edge_ids
        edge_counts = {edge_id: 0 for edge_id in current_edge_ids}

        vehicle_nodes = [n for n in snapshot.get("nodes", []) if n.get("node_type") == 1]
        for node in vehicle_nodes:
            route = node.get("route_left", [])
            for edge_id in route:
                if edge_id in edge_counts:
                    edge_counts[edge_id] += 1

        counts = np.array(list(edge_counts.values()))
        logs = np.log1p(counts)
        per_snapshot_counts.append(counts)
        per_snapshot_logs.append(logs)

    # Flatten
    all_counts = np.concatenate(per_snapshot_counts)
    all_logs = np.concatenate(per_snapshot_logs)

    # === Raw stats ===
    stats_raw = {
        "feature": "edge_route_count",
        "type": "numeric",
        "count": len(all_counts),
        "mean": float(np.mean(all_counts)),
        "std": float(np.std(all_counts)),
        "min": float(np.min(all_counts)),
        "25%": float(np.percentile(all_counts, 25)),
        "median": float(np.median(all_counts)),
        "75%": float(np.percentile(all_counts, 75)),
        "max": float(np.max(all_counts)),
        "skewness": float(skew(all_counts)),
        "kurtosis": float(kurtosis(all_counts)),
        "num_missing": int(np.sum(np.isnan(all_counts))),
        "num_unique": int(np.unique(all_counts).size),
        "normalization": "z-score"
    }

    # === Log stats ===
    stats_log = {
        "feature": "edge_route_count_log",
        "type": "numeric",
        "count": len(all_logs),
        "mean": float(np.mean(all_logs)),
        "std": float(np.std(all_logs)),
        "min": float(np.min(all_logs)),
        "25%": float(np.percentile(all_logs, 25)),
        "median": float(np.median(all_logs)),
        "75%": float(np.percentile(all_logs, 75)),
        "max": float(np.max(all_logs)),
        "skewness": float(skew(all_logs)),
        "kurtosis": float(kurtosis(all_logs)),
        "num_missing": int(np.sum(np.isnan(all_logs))),
        "num_unique": int(np.unique(all_logs).size),
        "normalization": "log"
    }

    # Save stats
    stats_df = pd.DataFrame([stats_raw, stats_log])
    stats_path = os.path.join(export_folder, "edge_route_count_summary.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"✅ Saved edge route counts summary to {stats_path}")

    # Optional: histograms and boxplots
    if plot_histogram:
        plt.figure()
        plt.hist(all_counts, bins=50)
        plt.title("Histogram of edge_route_count")
        plt.xlabel("Vehicle count per edge")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(export_folder, "edge_route_count_histogram.png"))
        plt.close()

        plt.figure()
        plt.hist(all_logs, bins=50)
        plt.title("Histogram of edge_route_count_log")
        plt.xlabel("log(1 + vehicle count)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(export_folder, "edge_route_count_log_histogram.png"))
        plt.close()

        plt.figure()
        plt.boxplot(all_counts, vert=False)
        plt.title("Boxplot of edge_route_count")
        plt.xlabel("Vehicle count per edge")
        plt.savefig(os.path.join(export_folder, "edge_route_count_boxplot.png"))
        plt.close()

        plt.figure()
        plt.boxplot(all_logs, vert=False)
        plt.title("Boxplot of edge_route_count_log")
        plt.xlabel("log(1 + vehicle count)")
        plt.savefig(os.path.join(export_folder, "edge_route_count_log_boxplot.png"))
        plt.close()

        print(f"📊 Saved visualizations to {export_folder}/")


def main():
    parser = argparse.ArgumentParser(description="Traffic Simulation EDA Toolset")
    parser.add_argument(
        "snapshots_folder",
        type=str,
        nargs="?",
        default="/media/guy/StorageVolume/traffic_data",
        help="Path to folder with snapshot JSON files (default: /media/guy/StorageVolume/traffic_data)"
    )
    parser.add_argument(
        "--labels_folder",
        type=str,
        default="/media/guy/StorageVolume/traffic_data/labels",
        help="Optional path to folder with label JSON files (e.g., labels_*.json)"
    )
    args = parser.parse_args()

    print("\nWelcome to the Dynamic Traffic Simulation EDA Toolkit")
    print("=====================================================")

    while True:
        print("\nMain Menu:")
        main_options = [
            "Summarize All Input Features for Preprocessing",
            "Summarize Label Features for Preprocessing",
            "Analyze edge route counts for Preprocessing",
            "Analyze Feature Distribution",
            "plot_route_distances_over_time",
            "plot_travel_durations_over_time",
            "Exit"
        ]
        main_choice = print_menu(main_options, "Choose action")

        if main_choice == len(main_options)-1:  # Exit
            print("Goodbye!")
            sys.exit(0)

        elif main_choice == 0:  # Summarize all features
            print("\nGenerating comprehensive summary for all features. This may take a few minutes...")
            summarize_features_for_preprocessing(args.snapshots_folder)
            print("Feature summaries exported to ./eda_exports/")
            continue

        elif main_choice == 1:  # Summarize labels
            print("\nGenerating label feature summary. This may take a few minutes...")
            if args.labels_folder:
                summarize_labels(args.labels_folder)
                print("Label summaries exported to ./eda_exports/")
            else:
                print("No labels folder specified. Skipping label summary.")
            continue
        elif main_choice == 2:  # Analyze edge route counts
            print("\nAnalyzing edge route counts for preprocessing...")
            analyze_edge_route_counts(args.snapshots_folder)
            print("Edge route counts analysis complete. Check ./eda_exports/")
            continue
        elif main_choice == 3: # Analyze Feature Distribution
            # Entity selection
            print("\nSelect entity for feature analysis:")
            entity_options = [
                "Vehicle features",
                "Junction features",
                "Road features (edges)"
            ]
            entity_map = ["vehicle", "junction", "edge"]
            entity_choice = print_menu(entity_options, "Choose entity type")
            entity_type = entity_map[entity_choice]

            features = get_available_features(args.snapshots_folder, entity_type)
            if not features:
                print("No features found for this entity.")
                continue

            print("\nAvailable Features:")
            feature_idx = print_menu(features, "Select a feature to analyze")
            feature_name = features[feature_idx]

            analysis_options = [
                "Show Histogram",
                "Show Boxplot",
                "Print Statistics",
                "Detect Outliers",
                "Show Normalization Preview",
                "Show Skewness Plot",
                "Back to Main Menu"
            ]
            while True:
                print(f"\nAnalysis Options for '{feature_name}':")
                analysis_idx = print_menu(analysis_options, "Select analysis output")
                if analysis_options[analysis_idx] == "Back to Main Menu":
                    break

                submenu_map = {
                    0: ["histogram"],
                    1: ["boxplot"],
                    2: ["stats"],
                    3: ["outliers"],
                    4: ["normalization_preview"],
                    5: ["skewness"],
                }
                analyze_feature_distribution(
                    snapshots_folder=args.snapshots_folder,
                    feature_name=feature_name,
                    entity_type=entity_type,
                    output_options=submenu_map[analysis_idx]
                )
        elif main_choice == 4:
            labels_file = os.path.join(args.snapshots_folder, "labels.json")
            plot_route_distances_over_time(labels_file)
        
        elif main_choice == 5:
            labels_file = os.path.join(args.snapshots_folder, "labels.json")
            plot_travel_durations_over_time(labels_file)
        
        print("\nWould you like to:")
        next_step = print_menu(["Start Over", "Exit"])
        if next_step == 1:
            print("Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
