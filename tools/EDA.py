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

def summarize_features_for_preprocessing(
    snapshots_folder,
    export_folder="./eda_exports"
):
    def _safe_filename(s):
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

    def _summarize_features(values):
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
                        feature_values.setdefault(k, []).append(v)
        rows = []
        for feat, vals in feature_values.items():
            # Skip features that are lists or dicts (complex structures)
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
                summary = _summarize_features(vals_for_stats)
            else:
                # Only keep scalar (hashable) values for categorical stats
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

def main():
    parser = argparse.ArgumentParser(description="Traffic Simulation EDA Toolset")
    parser.add_argument(
        "snapshots_folder",
        type=str,
        nargs="?",
        default="/media/guy/StorageVolume/traffic_data",
        help="Path to folder with snapshot JSON files (default: /media/guy/StorageVolume/traffic_data)"
    )
    args = parser.parse_args()

    print("\nWelcome to the Dynamic Traffic Simulation EDA Toolkit")
    print("=====================================================")

    while True:
        print("\nMain Menu:")
        main_options = [
            "Analyze Feature Distribution",
            "Summarize All Features for Preprocessing",
            "Exit"
        ]
        main_choice = print_menu(main_options, "Choose action")

        if main_choice == 2:  # Exit
            print("Goodbye!")
            sys.exit(0)

        elif main_choice == 1:  # Summarize all features
            print("\nGenerating comprehensive summary for all features. This may take a few minutes...")
            summarize_features_for_preprocessing(args.snapshots_folder)
            print("Feature summaries exported to ./eda_exports/")
            continue

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

        print("\nWould you like to:")
        next_step = print_menu(["Start Over", "Exit"])
        if next_step == 1:
            print("Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
