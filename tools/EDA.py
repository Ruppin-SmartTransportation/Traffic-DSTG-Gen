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
import gc
import psutil

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(stage=""):
    """Log current memory usage"""
    memory_mb = get_memory_usage()
    print(f"💾 Memory usage {stage}: {memory_mb:.1f} MB")
    return memory_mb

def safe_filename(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def get_snapshot_files(snapshots_folder):
    """Return all snapshot files (excluding any 'labels' file) for memory-efficient processing."""
    all_files = [
        os.path.join(snapshots_folder, f)
        for f in os.listdir(snapshots_folder)
        if f.endswith('.json') and "labels" not in f
    ]
    print(f"📁 Found {len(all_files):,} snapshot files for analysis")
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
    Memory-efficient chunked processing for large datasets.
    """
    os.makedirs(export_folder, exist_ok=True)
    all_files = get_snapshot_files(snapshots_folder)
    
    if not all_files:
        print("No snapshot files found.")
        return
    
    print(f"⚠️  Large dataset detected ({len(all_files):,} files). Using memory-efficient processing...")
    
    # Process in chunks to avoid memory issues
    chunk_size = 50  # Process 50 files at a time (reduced for memory efficiency)
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
    
    values = []
    
    for chunk_idx in tqdm(range(total_chunks), desc=f"Processing chunks for {feature_name} ({entity_type})", unit="chunk", position=0):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_files))
        chunk_files = all_files[start_idx:end_idx]
        
        # Process chunk
        chunk_values = []
        for file in tqdm(chunk_files, desc=f"Chunk {chunk_idx+1}/{total_chunks}", leave=False, unit="file"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if entity_type == "vehicle":
                        for node in data.get("nodes", []):
                            if node.get("node_type") == 1:
                                v = node.get(feature_name)
                                if v is not None:
                                    chunk_values.append(v)
                    elif entity_type == "junction":
                        for node in data.get("nodes", []):
                            if node.get("node_type") == 0:
                                v = node.get(feature_name)
                                if v is not None:
                                    chunk_values.append(v)
                    elif entity_type == "edge":
                        for edge in data.get("edges", []):
                            v = edge.get(feature_name)
                            if v is not None:
                                chunk_values.append(v)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load {file}: {str(e)}")
                continue
        
        values.extend(chunk_values)
        
        # Apply sampling within chunk if requested
        if sample_size and len(values) > sample_size:
            values = random.sample(values, sample_size)
            print(f"📊 Applied sampling: using {sample_size:,} values")
            break
    
    if not values:
        print(f"No values found for feature '{feature_name}'.")
        return

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

def summarize_labels(labels_folder, export_folder="./eda_exports", entity_type=None):
    print("\nProcessing label files...")
    os.makedirs(export_folder, exist_ok=True)
    all_files = [
        os.path.join(labels_folder, f)
        for f in os.listdir(labels_folder)
        if f.startswith("labels_") and f.endswith(".json")
    ]
    
    if not all_files:
        print("No label files found.")
        return
    
    print(f"⚠️  Large dataset detected ({len(all_files):,} files). Using memory-efficient processing...")

    feature_values = dict()
    
    # Process in chunks to avoid memory issues
    chunk_size = 50  # Process 50 files at a time (reduced for memory efficiency)
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(total_chunks), desc="Scanning label files", unit="chunk"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_files))
        chunk_files = all_files[start_idx:end_idx]
        
        # Process chunk
        for file in tqdm(chunk_files, desc=f"Chunk {chunk_idx+1}/{total_chunks}", leave=False, unit="file"):
            try:
                with open(file, 'r') as f:
                    label_list = json.load(f)
                    for entry in label_list:
                        for k, v in entry.items():
                            feature_values.setdefault(k, []).append(v)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load {file}: {str(e)}")
                continue

    rows = []
    for feat, vals in feature_values.items():
        # Force certain features to be categorical
        force_categorical = False
        if feat == 'num_lanes':
            force_categorical = True
        if feat == 'speed' and entity_type == 'edge':
            force_categorical = True
        if feat == 'node_type':
            force_categorical = True
        if all(isinstance(v, (list, dict)) or v is None for v in vals):
            print(f"Skipping complex feature '{feat}' (all values are lists/dicts).")
            continue
        vals_clean = [v if v not in [None, "", "None"] else np.nan for v in vals]
        try:
            vals_num = pd.to_numeric(pd.Series(vals_clean), errors='coerce')
            is_numeric = not np.all(np.isnan(vals_num))
        except Exception:
            is_numeric = False
        # If forced categorical, treat as categorical
        if force_categorical:
            keys = sorted(set([v for v in vals_clean if v not in [None, '', 'None', np.nan]]))
            summary = {
                "feature": feat,
                "type": "categorical",
                "count": len(vals_clean),
                "num_missing": sum(pd.isna(vals_clean)),
                "num_unique": len(keys),
                "keys": keys
            }
        elif is_numeric:
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
                "97%": np.percentile(vals_for_stats, 97),
                "98%": np.percentile(vals_for_stats, 98),
                "99%": np.percentile(vals_for_stats, 99),
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
                "num_unique": len(set(vals_clean)),
                "keys": sorted(set([v for v in vals_clean if v not in [None, '', 'None', np.nan]]))
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
    import gc

    def _safe_filename(s):
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

    def _update_numeric_stats(stats_dict, feature_name, value):
        """Update running statistics for numeric features"""
        if feature_name not in stats_dict:
            stats_dict[feature_name] = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min_val': float('inf'),
                'max_val': float('-inf'),
                'values': []  # Only keep for small datasets
            }
        
        stats = stats_dict[feature_name]
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_sq'] += value * value
        stats['min_val'] = min(stats['min_val'], value)
        stats['max_val'] = max(stats['max_val'], value)
        
        # Only keep values for small datasets (less than 10000 values)
        if stats['count'] <= 10000:
            stats['values'].append(value)

    def _update_categorical_stats(stats_dict, feature_name, value):
        """Update running statistics for categorical features"""
        if feature_name not in stats_dict:
            stats_dict[feature_name] = {
                'count': 0,
                'unique_values': set(),
                'value_counts': {}
            }
        
        stats = stats_dict[feature_name]
        stats['count'] += 1
        stats['unique_values'].add(value)
        stats['value_counts'][value] = stats['value_counts'].get(value, 0) + 1

    def _finalize_numeric_stats(stats_dict, feature_name):
        """Convert running statistics to final summary"""
        stats = stats_dict[feature_name]
        if stats['count'] == 0:
            return None
            
        mean_val = stats['sum'] / stats['count']
        variance = (stats['sum_sq'] / stats['count']) - (mean_val * mean_val)
        std_val = np.sqrt(max(0, variance))
        
        # Calculate percentiles and skewness/kurtosis if we have values
        if stats['values']:
            values = np.array(stats['values'])
            q25 = np.percentile(values, 25)
            median = np.median(values)
            q75 = np.percentile(values, 75)
            skewness = skew(values) if len(values) > 2 else 0
            kurt = kurtosis(values) if len(values) > 2 else 0
        else:
            # For large datasets, estimate percentiles
            q25 = median = q75 = mean_val
            skewness = kurt = 0
        
        return {
            "count": stats['count'],
            "mean": mean_val,
            "std": std_val,
            "min": stats['min_val'],
            "25%": q25,
            "median": median,
            "75%": q75,
            "max": stats['max_val'],
            "skewness": skewness,
            "kurtosis": kurt,
            "num_missing": 0,  # We're not tracking missing values in this approach
            "num_unique": len(stats['values']) if stats['values'] else 0,
            "normalization": "z-score" if (
                feature_name.endswith("_log") or
                feature_name in ["speed", "length", "avg_speed"]
            ) else None
        }

    def _finalize_categorical_stats(stats_dict, feature_name):
        """Convert running statistics to final summary"""
        stats = stats_dict[feature_name]
        if stats['count'] == 0:
            return None
            
        return {
            "count": stats['count'],
            "num_missing": 0,  # We're not tracking missing values in this approach
            "num_unique": len(stats['unique_values']),
            "value_counts": stats['value_counts'] if len(stats['unique_values']) < 20 else None
        }

    os.makedirs(export_folder, exist_ok=True)
    all_files = get_snapshot_files(snapshots_folder)
    
    if not all_files:
        print("No snapshot files found.")
        return
    
    print(f"⚠️  Large dataset detected ({len(all_files):,} files). Using memory-efficient processing...")

    entity_types = {
        "vehicle": lambda node: node.get("node_type") == 1,
        "junction": lambda node: node.get("node_type") == 0,
        "edge": None,  # For "edges" array
    }

    for entity, filter_fn in entity_types.items():
        print(f"\nProcessing {entity} features...")
        log_memory_usage(f"before {entity} processing")
        
        # Initialize running statistics dictionaries
        numeric_stats = {}
        categorical_stats = {}
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 25  # Reduced chunk size for better memory management
        total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(total_chunks), desc=f"Scanning {entity} files", unit="chunk", position=0):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_files))
            chunk_files = all_files[start_idx:end_idx]
            
            # Process chunk
            for file in tqdm(chunk_files, desc=f"Files in chunk {chunk_idx+1}/{total_chunks}", leave=False, unit="file", position=1):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if entity == "edge":
                            items = data.get("edges", [])
                        else:
                            items = [n for n in data.get("nodes", []) if filter_fn(n)]
                        
                        for item in items:
                            for k, v in item.items():
                                # Skip complex structures
                                if isinstance(v, (list, dict)) and k != "vehicles_on_road":
                                    continue
                                
                                # Special handling for vehicles_on_road → count only
                                if k == "vehicles_on_road":
                                    count = len(v) if isinstance(v, list) else 0
                                    _update_numeric_stats(numeric_stats, "vehicles_on_road_count", count)
                                    _update_numeric_stats(numeric_stats, "vehicles_on_road_count_log", np.log1p(count))
                                    continue
                                
                                # Handle None/empty values
                                if v in [None, "", "None"]:
                                    continue
                                
                                # Try to convert to numeric
                                try:
                                    num_val = float(v)
                                    _update_numeric_stats(numeric_stats, k, num_val)
                                except (ValueError, TypeError):
                                    # Treat as categorical
                                    _update_categorical_stats(categorical_stats, k, str(v))
                                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️  Warning: Could not load {file}: {str(e)}")
                    continue
            
            # Force garbage collection after each chunk
            gc.collect()
            
            # Log memory usage every 10 chunks
            if (chunk_idx + 1) % 10 == 0:
                log_memory_usage(f"after chunk {chunk_idx + 1}/{total_chunks}")

        # Finalize statistics
        rows = []
        
        # Process numeric features
        for feat, stats in numeric_stats.items():
            summary = _finalize_numeric_stats(numeric_stats, feat)
            if summary:
                row = {"feature": feat, "type": "numeric"}
                row.update(summary)
                rows.append(row)
        
        # Process categorical features
        for feat, stats in categorical_stats.items():
            summary = _finalize_categorical_stats(categorical_stats, feat)
            if summary:
                row = {"feature": feat, "type": "categorical"}
                row.update(summary)
                rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        outpath = os.path.join(export_folder, f"{entity}_feature_summary.csv")
        df.to_csv(outpath, index=False)
        print(f"Saved {entity} summary to {outpath}")
        
        # Clear memory
        del numeric_stats, categorical_stats, df
        gc.collect()
        log_memory_usage(f"after {entity} processing")

def print_menu(options, prompt="Select an option:"):
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        choice = input(f"{prompt} (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        else:
            print("Invalid choice. Try again.")

def analyze_eta_categories(labels_folder, export_folder="./eda_exports", create_main_csv=True):
    """
    Analyze ETA values and suggest category thresholds for balanced short/medium/long classification.
    
    This function analyzes the distribution of ETA values from all label files in the labels folder
    and suggests optimal thresholds to create a balanced dataset with three categories: 
    short, medium, and long ETA.
    
    Args:
        labels_folder (str): Path to the folder containing label JSON files (labels_*.json)
        export_folder (str): Folder to export analysis results
    """
    print("🔍 Analyzing ETA values for category classification...")
    
    # Check if labels folder exists
    if not os.path.exists(labels_folder):
        print(f"❌ Labels folder not found: {labels_folder}")
        return
    
    # Get all label files
    label_files = []
    for file in os.listdir(labels_folder):
        if file.startswith("labels_") and file.endswith(".json"):
            label_files.append(os.path.join(labels_folder, file))
    
    if not label_files:
        print(f"❌ No label files found in {labels_folder}")
        print("Expected files: labels_*.json")
        return
    
    print(f"📁 Found {len(label_files)} label files")
    
    # Load and combine all labels data
    all_labels = []
    print(f"📊 Processing {len(label_files)} label files...")
    
    # Process in chunks to avoid memory issues
    chunk_size = 50  # Process 50 files at a time
    total_chunks = (len(label_files) + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(total_chunks), desc="Loading label files", unit="chunk", position=0):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(label_files))
        chunk_files = label_files[start_idx:end_idx]
        
        # Process chunk
        for label_file in tqdm(chunk_files, desc=f"Files in chunk {chunk_idx+1}/{total_chunks}", leave=False, unit="file", position=1):
            try:
                with open(label_file, 'r') as f:
                    file_labels = json.load(f)
                    if isinstance(file_labels, list):
                        all_labels.extend(file_labels)
                    else:
                        print(f"⚠️  Warning: {label_file} does not contain a list of labels")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load {label_file}: {str(e)}")
                continue
    
    # Extract ETA values with progress bar
    eta_values = []
    print(f"📊 Processing {len(all_labels):,} label entries...")
    for entry in tqdm(all_labels, desc="Extracting ETA values", unit="entries"):
        eta = entry.get('eta')
        if eta is not None and isinstance(eta, (int, float)):
            eta_values.append(eta)
    
    if not eta_values:
        print("❌ No valid ETA values found in labels file.")
        return
    
    eta_values = np.array(eta_values)
    print(f"📊 Found {len(eta_values)} valid ETA values")
    
    # Basic statistics
    stats = {
        'count': len(eta_values),
        'mean': np.mean(eta_values),
        'std': np.std(eta_values),
        'min': np.min(eta_values),
        'max': np.max(eta_values),
        'median': np.median(eta_values),
        '25th_percentile': np.percentile(eta_values, 25),
        '75th_percentile': np.percentile(eta_values, 75),
        'skewness': skew(eta_values),
        'kurtosis': kurtosis(eta_values)
    }
    
    print(f"\n📈 ETA Statistics:")
    print(f"   Count: {stats['count']:,}")
    print(f"   Mean: {stats['mean']:.1f} seconds ({stats['mean']/60:.1f} minutes)")
    print(f"   Std: {stats['std']:.1f} seconds ({stats['std']/60:.1f} minutes)")
    print(f"   Min: {stats['min']:.1f} seconds ({stats['min']/60:.1f} minutes)")
    print(f"   Max: {stats['max']:.1f} seconds ({stats['max']/60:.1f} minutes)")
    print(f"   Median: {stats['median']:.1f} seconds ({stats['median']/60:.1f} minutes)")
    print(f"   25th percentile: {stats['25th_percentile']:.1f} seconds ({stats['25th_percentile']/60:.1f} minutes)")
    print(f"   75th percentile: {stats['75th_percentile']:.1f} seconds ({stats['75th_percentile']/60:.1f} minutes)")
    print(f"   Skewness: {stats['skewness']:.3f}")
    print(f"   Kurtosis: {stats['kurtosis']:.3f}")
    
    # Create visualizations
    os.makedirs(export_folder, exist_ok=True)
    
    print("📊 Creating visualizations...")
    
    # 1. Histogram with current distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(eta_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('ETA Distribution (All Values)')
    plt.xlabel('ETA (seconds)')
    plt.ylabel('Frequency')
    plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.1f}s')
    plt.axvline(stats['median'], color='orange', linestyle='--', label=f'Median: {stats["median"]:.1f}s')
    plt.legend()
    
    # 2. Log-scale histogram
    plt.subplot(2, 3, 2)
    plt.hist(eta_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.yscale('log')
    plt.title('ETA Distribution (Log Scale)')
    plt.xlabel('ETA (seconds)')
    plt.ylabel('Frequency (log)')
    
    # 3. Box plot
    plt.subplot(2, 3, 3)
    plt.boxplot(eta_values, vert=False)
    plt.title('ETA Box Plot')
    plt.xlabel('ETA (seconds)')
    
    # 4. Cumulative distribution
    plt.subplot(2, 3, 4)
    sorted_etas = np.sort(eta_values)
    cumulative = np.arange(1, len(sorted_etas) + 1) / len(sorted_etas)
    plt.plot(sorted_etas, cumulative, linewidth=2)
    plt.title('Cumulative Distribution')
    plt.xlabel('ETA (seconds)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # 5. Distribution by time of day (if origin_time_sec is available)
    plt.subplot(2, 3, 5)
    origin_times = []
    print("🕐 Extracting origin times for temporal analysis...")
    for entry in tqdm(all_labels, desc="Processing origin times", unit="entries"):
        origin_time = entry.get('origin_time_sec')
        if origin_time is not None:
            origin_times.append(origin_time)
    
    if origin_times:
        # Convert to hours of day
        hours_of_day = [(t % 86400) / 3600 for t in origin_times]
        plt.scatter(hours_of_day, eta_values[:len(hours_of_day)], alpha=0.5, s=1)
        plt.title('ETA vs Time of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('ETA (seconds)')
        plt.xlim(0, 24)
    else:
        plt.text(0.5, 0.5, 'No origin time data\navailable', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('ETA vs Time of Day')
    
    # 6. Distribution by day of week (if origin_time_sec is available)
    plt.subplot(2, 3, 6)
    if origin_times:
        # Convert to day of week (0=Monday, 6=Sunday)
        days_of_week = [(t // 86400) % 7 for t in origin_times]
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Calculate median ETA for each day
        day_medians = []
        for day in range(7):
            day_mask = [d == day for d in days_of_week[:len(eta_values)]]
            if any(day_mask):
                day_medians.append(np.median(eta_values[day_mask]))
            else:
                day_medians.append(0)
        
        plt.bar(day_names, day_medians, color='lightcoral')
        plt.title('Median ETA by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Median ETA (seconds)')
    else:
        plt.text(0.5, 0.5, 'No origin time data\navailable', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Median ETA by Day of Week')
    
    plt.tight_layout()
    plt.savefig(os.path.join(export_folder, 'eta_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Suggest category thresholds
    print(f"\n🎯 Category Threshold Suggestions:")
    
    # Method 1: Equal thirds (33.33% each)
    third_33 = np.percentile(eta_values, 33.33)
    third_67 = np.percentile(eta_values, 66.67)
    
    # Method 2: Equal thirds (33.33% each) - alternative
    third_33_alt = np.percentile(eta_values, 33.33)
    third_67_alt = np.percentile(eta_values, 66.67)
    
    # Method 3: Based on natural breaks (using percentiles)
    p25 = np.percentile(eta_values, 25)
    p75 = np.percentile(eta_values, 75)
    
    # Method 4: Based on mean plus minus 0.5 std
    mean_minus_half_std = max(0, stats['mean'] - 0.5 * stats['std'])
    mean_plus_half_std = stats['mean'] + 0.5 * stats['std']
    
    # Method 5: Based on median plus minus 0.5 IQR
    iqr = stats['75th_percentile'] - stats['25th_percentile']
    median_minus_half_iqr = max(0, stats['median'] - 0.5 * iqr)
    median_plus_half_iqr = stats['median'] + 0.5 * iqr
    
    # Calculate category distributions for each method
    def calculate_category_distribution(threshold1, threshold2):
        short = np.sum(eta_values < threshold1)
        medium = np.sum((eta_values >= threshold1) & (eta_values < threshold2))
        long = np.sum(eta_values >= threshold2)
        total = len(eta_values)
        return {
            'short': {'count': short, 'percentage': short/total*100},
            'medium': {'count': medium, 'percentage': medium/total*100},
            'long': {'count': long, 'percentage': long/total*100}
        }
    
    methods = {
        'Equal Thirds (33.33%)': (third_33, third_67),
        'Quartile-based (25-75)': (p25, p75),
        'Mean plus minus 0.5 Std': (mean_minus_half_std, mean_plus_half_std),
        'Median plus minus 0.5 IQR': (median_minus_half_iqr, median_plus_half_iqr)
    }
    
    print(f"\n📊 Category Distribution Analysis:")
    print(f"{'Method':<25} {'Short':<15} {'Medium':<15} {'Long':<15}")
    print("-" * 70)
    
    best_method = None
    best_balance = float('inf')
    
    print("🔍 Evaluating threshold methods...")
    for method_name, (t1, t2) in tqdm(methods.items(), desc="Analyzing methods", unit="method"):
        dist = calculate_category_distribution(t1, t2)
        balance_score = abs(dist['short']['percentage'] - 33.33) + abs(dist['medium']['percentage'] - 33.33) + abs(dist['long']['percentage'] - 33.33)
        
        print(f"{method_name:<25} {dist['short']['percentage']:>6.1f}% ({dist['short']['count']:>6,}) {dist['medium']['percentage']:>6.1f}% ({dist['medium']['count']:>6,}) {dist['long']['percentage']:>6.1f}% ({dist['long']['count']:>6,})")
        
        if balance_score < best_balance:
            best_balance = balance_score
            best_method = method_name
    
    # Recommend the best method
    best_thresholds = methods[best_method]
    best_dist = calculate_category_distribution(*best_thresholds)
    
    print(f"\n🏆 RECOMMENDED METHOD: {best_method}")
    print(f"   Short ETA:  < {best_thresholds[0]:.1f} seconds ({best_thresholds[0]/60:.1f} minutes)")
    print(f"   Medium ETA: {best_thresholds[0]:.1f} - {best_thresholds[1]:.1f} seconds ({best_thresholds[0]/60:.1f} - {best_thresholds[1]/60:.1f} minutes)")
    print(f"   Long ETA:   ≥ {best_thresholds[1]:.1f} seconds ({best_thresholds[1]/60:.1f} minutes)")
    
    print("🎯 Creating category analysis visualization...")
    
    # Create category visualization
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with category boundaries
    plt.subplot(2, 2, 1)
    plt.hist(eta_values, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(best_thresholds[0], color='red', linestyle='--', linewidth=2, label=f'Short/Medium: {best_thresholds[0]:.1f}s')
    plt.axvline(best_thresholds[1], color='red', linestyle='--', linewidth=2, label=f'Medium/Long: {best_thresholds[1]:.1f}s')
    plt.title(f'ETA Distribution with {best_method} Thresholds')
    plt.xlabel('ETA (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Category pie chart
    plt.subplot(2, 2, 2)
    categories = ['Short', 'Medium', 'Long']
    sizes = [best_dist['short']['count'], best_dist['medium']['count'], best_dist['long']['count']]
    colors = ['lightgreen', 'orange', 'red']
    plt.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Category Distribution')
    
    # Category box plots
    plt.subplot(2, 2, 3)
    short_etas = eta_values[eta_values < best_thresholds[0]]
    medium_etas = eta_values[(eta_values >= best_thresholds[0]) & (eta_values < best_thresholds[1])]
    long_etas = eta_values[eta_values >= best_thresholds[1]]
    
    category_data = [short_etas, medium_etas, long_etas]
    plt.boxplot(category_data, labels=categories)
    plt.title('ETA Distribution by Category')
    plt.ylabel('ETA (seconds)')
    
    # Category statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""Category Statistics:
    
Short ETA (< {best_thresholds[0]:.1f}s):
  Count: {best_dist['short']['count']:,} ({best_dist['short']['percentage']:.1f}%)
  Mean: {np.mean(short_etas):.1f}s
  Median: {np.median(short_etas):.1f}s

Medium ETA ({best_thresholds[0]:.1f}s - {best_thresholds[1]:.1f}s):
  Count: {best_dist['medium']['count']:,} ({best_dist['medium']['percentage']:.1f}%)
  Mean: {np.mean(medium_etas):.1f}s
  Median: {np.median(medium_etas):.1f}s

Long ETA (≥ {best_thresholds[1]:.1f}s):
  Count: {best_dist['long']['count']:,} ({best_dist['long']['percentage']:.1f}%)
  Mean: {np.mean(long_etas):.1f}s
  Median: {np.median(long_etas):.1f}s"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(export_folder, 'eta_category_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to JSON
    results = {
        'dataset_info': {
            'total_samples': len(eta_values),
            'eta_range_seconds': [float(stats['min']), float(stats['max'])],
            'eta_range_minutes': [float(stats['min']/60), float(stats['max']/60)]
        },
        'statistics': {k: float(v) for k, v in stats.items()},
        'recommended_thresholds': {
            'method': best_method,
            'short_threshold_seconds': float(best_thresholds[0]),
            'short_threshold_minutes': float(best_thresholds[0]/60),
            'long_threshold_seconds': float(best_thresholds[1]),
            'long_threshold_minutes': float(best_thresholds[1]/60)
        },
        'category_distribution': {
            'short': {
                'count': int(best_dist['short']['count']),
                'percentage': float(best_dist['short']['percentage']),
                'mean_seconds': float(np.mean(short_etas)),
                'median_seconds': float(np.median(short_etas))
            },
            'medium': {
                'count': int(best_dist['medium']['count']),
                'percentage': float(best_dist['medium']['percentage']),
                'mean_seconds': float(np.mean(medium_etas)),
                'median_seconds': float(np.median(medium_etas))
            },
            'long': {
                'count': int(best_dist['long']['count']),
                'percentage': float(best_dist['long']['percentage']),
                'mean_seconds': float(np.mean(long_etas)),
                'median_seconds': float(np.median(long_etas))
            }
        },
        'all_methods': {}
    }
    
    # Add all methods for comparison
    print("💾 Saving detailed results...")
    for method_name, (t1, t2) in tqdm(methods.items(), desc="Saving method data", unit="method"):
        dist = calculate_category_distribution(t1, t2)
        results['all_methods'][method_name] = {
            'short_threshold_seconds': float(t1),
            'long_threshold_seconds': float(t2),
            'distribution': {k: {'count': int(v['count']), 'percentage': float(v['percentage'])} for k, v in dist.items()}
        }
    
    # Save results
    results_path = os.path.join(export_folder, 'eta_category_analysis.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comprehensive CSV output
    if create_main_csv:
        print("📊 Creating CSV output with all analyzed data...")
        create_csv_output(eta_values, results, all_labels, export_folder)
    else:
        print("⏭️  Skipping main CSV creation (use create_main_csv=True to enable)")
        # Still create summary CSVs
        create_summary_csvs(results, eta_values, export_folder)
    
    print(f"\n💾 Analysis results saved to:")
    print(f"   📊 Overview plots: {export_folder}/eta_analysis_overview.png")
    print(f"   🎯 Category analysis: {export_folder}/eta_category_analysis.png")
    print(f"   📋 Detailed results: {export_folder}/eta_category_analysis.json")
    print(f"   📊 CSV data: {export_folder}/eta_analysis_complete.csv")
    
    return results

def create_csv_output(eta_values, results, all_labels, export_folder):
    """
    Create comprehensive CSV output with all analyzed ETA data and category distributions.
    Memory-optimized for large datasets.
    
    Args:
        eta_values (np.array): Array of all ETA values
        results (dict): Analysis results dictionary
        all_labels (list): All label entries
        export_folder (str): Folder to save CSV file
    """
    import pandas as pd
    
    # Get recommended thresholds
    recommended = results['recommended_thresholds']
    short_threshold = recommended['short_threshold_seconds']
    long_threshold = recommended['long_threshold_seconds']
    
    print("📝 Processing individual ETA entries...")
    print(f"⚠️  Large dataset detected ({len(all_labels):,} entries). Using memory-efficient processing...")
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000  # Process 100000 entries at a time
    total_chunks = (len(all_labels) + chunk_size - 1) // chunk_size
    
    # Create CSV file and write header
    csv_path = os.path.join(export_folder, 'eta_analysis_complete.csv')
    header_written = False
    
    valid_entries = 0
    
    for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks", unit="chunk"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_labels))
        chunk_entries = all_labels[start_idx:end_idx]
        
        # Process chunk
        chunk_data = []
        for i, entry in enumerate(chunk_entries):
            global_idx = start_idx + i
            eta = entry.get('eta')
            if eta is not None and isinstance(eta, (int, float)):
                # Determine category
                if eta < short_threshold:
                    eta_category = "short"
                    eta_category_code = 0
                elif eta < long_threshold:
                    eta_category = "medium"
                    eta_category_code = 1
                else:
                    eta_category = "long"
                    eta_category_code = 2
                
                # Create row data
                origin_time_sec = entry.get('origin_time_sec')
                row = {
                    'vehicle_id': entry.get('vehicle_id', f'veh_{global_idx}'),
                    'eta_seconds': float(eta),
                    'eta_minutes': float(eta / 60),
                    'eta_category': eta_category,
                    'eta_category_code': eta_category_code,
                    'origin_time_sec': origin_time_sec,
                    'destination_time_sec': entry.get('destination_time_sec', None),
                    'total_travel_time_seconds': entry.get('total_travel_time_seconds', None),
                    'origin_time_hours': origin_time_sec / 3600 if origin_time_sec else None,
                    'day_of_week': (origin_time_sec // 86400) % 7 if origin_time_sec else None,
                    'hour_of_day': (origin_time_sec % 86400) / 3600 if origin_time_sec else None
                }
                chunk_data.append(row)
                valid_entries += 1
        
        # Convert chunk to DataFrame and write to CSV
        if chunk_data:
            chunk_df = pd.DataFrame(chunk_data)
            chunk_df.to_csv(csv_path, mode='w' if not header_written else 'a', 
                           header=not header_written, index=False)
            header_written = True
            
            # Clear chunk data to free memory
            del chunk_data, chunk_df
    
    print(f"✅ Processed {valid_entries:,} valid entries")
    
    # Create main DataFrame for summary (sample only for large datasets)
    if valid_entries > 1000000:  # If more than 1M entries, use sample for display
        print("📊 Large dataset detected. Using sample for preview...")
        sample_df = pd.read_csv(csv_path, nrows=5)
    else:
        sample_df = pd.read_csv(csv_path)
    
    # Add summary statistics
    print("📊 Adding summary statistics...")
    
    # Basic statistics
    stats_summary = {
        'metric': [
            'total_samples',
            'mean_eta_seconds',
            'mean_eta_minutes', 
            'std_eta_seconds',
            'std_eta_minutes',
            'min_eta_seconds',
            'min_eta_minutes',
            'max_eta_seconds',
            'max_eta_minutes',
            'median_eta_seconds',
            'median_eta_minutes',
            '25th_percentile_seconds',
            '25th_percentile_minutes',
            '75th_percentile_seconds',
            '75th_percentile_minutes',
            'skewness',
            'kurtosis'
        ],
        'value': [
            len(eta_values),
            float(np.mean(eta_values)),
            float(np.mean(eta_values) / 60),
            float(np.std(eta_values)),
            float(np.std(eta_values) / 60),
            float(np.min(eta_values)),
            float(np.min(eta_values) / 60),
            float(np.max(eta_values)),
            float(np.max(eta_values) / 60),
            float(np.median(eta_values)),
            float(np.median(eta_values) / 60),
            float(np.percentile(eta_values, 25)),
            float(np.percentile(eta_values, 25) / 60),
            float(np.percentile(eta_values, 75)),
            float(np.percentile(eta_values, 75) / 60),
            float(skew(eta_values)),
            float(kurtosis(eta_values))
        ]
    }
    
    # Category distribution
    recommended = results['recommended_thresholds']
    category_dist = results['category_distribution']
    
    category_summary = {
        'category': ['short', 'medium', 'long'],
        'threshold_seconds': [
            recommended['short_threshold_seconds'],
            f"{recommended['short_threshold_seconds']} - {recommended['long_threshold_seconds']}",
            recommended['long_threshold_seconds']
        ],
        'threshold_minutes': [
            recommended['short_threshold_minutes'],
            f"{recommended['short_threshold_minutes']:.1f} - {recommended['long_threshold_minutes']:.1f}",
            recommended['long_threshold_minutes']
        ],
        'count': [
            category_dist['short']['count'],
            category_dist['medium']['count'],
            category_dist['long']['count']
        ],
        'percentage': [
            category_dist['short']['percentage'],
            category_dist['medium']['percentage'],
            category_dist['long']['percentage']
        ],
        'mean_eta_seconds': [
            category_dist['short']['mean_seconds'],
            category_dist['medium']['mean_seconds'],
            category_dist['long']['mean_seconds']
        ],
        'mean_eta_minutes': [
            category_dist['short']['mean_seconds'] / 60,
            category_dist['medium']['mean_seconds'] / 60,
            category_dist['long']['mean_seconds'] / 60
        ],
        'median_eta_seconds': [
            category_dist['short']['median_seconds'],
            category_dist['medium']['median_seconds'],
            category_dist['long']['median_seconds']
        ],
        'median_eta_minutes': [
            category_dist['short']['median_seconds'] / 60,
            category_dist['medium']['median_seconds'] / 60,
            category_dist['long']['median_seconds'] / 60
        ]
    }
    
    # All method comparisons
    all_methods_data = []
    for method_name, method_data in results['all_methods'].items():
        dist = method_data['distribution']
        all_methods_data.append({
            'method': method_name,
            'short_threshold_seconds': method_data['short_threshold_seconds'],
            'long_threshold_seconds': method_data['long_threshold_seconds'],
            'short_threshold_minutes': method_data['short_threshold_seconds'] / 60,
            'long_threshold_minutes': method_data['long_threshold_seconds'] / 60,
            'short_count': dist['short']['count'],
            'short_percentage': dist['short']['percentage'],
            'medium_count': dist['medium']['count'],
            'medium_percentage': dist['medium']['percentage'],
            'long_count': dist['long']['count'],
            'long_percentage': dist['long']['percentage']
        })
    
    # Save summary statistics
    stats_df = pd.DataFrame(stats_summary)
    stats_path = os.path.join(export_folder, 'eta_analysis_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    
    # Save category summary
    category_df = pd.DataFrame(category_summary)
    category_path = os.path.join(export_folder, 'eta_analysis_categories.csv')
    category_df.to_csv(category_path, index=False)
    
    # Save all methods comparison
    methods_df = pd.DataFrame(all_methods_data)
    methods_path = os.path.join(export_folder, 'eta_analysis_methods.csv')
    methods_df.to_csv(methods_path, index=False)
    
    print(f"✅ CSV files created:")
    print(f"   📊 Main data: {csv_path} ({valid_entries:,} rows)")
    print(f"   📈 Statistics: {stats_path}")
    print(f"   🎯 Categories: {category_path}")
    print(f"   🔍 Methods: {methods_path}")
    
    # Print sample of the data
    print(f"\n📋 Sample of CSV data (first 5 rows):")
    print(sample_df.to_string(index=False))
    
    return sample_df

def create_summary_csvs(results, eta_values, export_folder):
    """
    Create summary CSV files without the main dataset (for memory efficiency).
    
    Args:
        results (dict): Analysis results dictionary
        eta_values (np.array): Array of all ETA values
        export_folder (str): Folder to save CSV files
    """
    import pandas as pd
    
    # Basic statistics
    stats_summary = {
        'metric': [
            'total_samples',
            'mean_eta_seconds',
            'mean_eta_minutes', 
            'std_eta_seconds',
            'std_eta_minutes',
            'min_eta_seconds',
            'min_eta_minutes',
            'max_eta_seconds',
            'max_eta_minutes',
            'median_eta_seconds',
            'median_eta_minutes',
            '25th_percentile_seconds',
            '25th_percentile_minutes',
            '75th_percentile_seconds',
            '75th_percentile_minutes',
            'skewness',
            'kurtosis'
        ],
        'value': [
            len(eta_values),
            float(np.mean(eta_values)),
            float(np.mean(eta_values) / 60),
            float(np.std(eta_values)),
            float(np.std(eta_values) / 60),
            float(np.min(eta_values)),
            float(np.min(eta_values) / 60),
            float(np.max(eta_values)),
            float(np.max(eta_values) / 60),
            float(np.median(eta_values)),
            float(np.median(eta_values) / 60),
            float(np.percentile(eta_values, 25)),
            float(np.percentile(eta_values, 25) / 60),
            float(np.percentile(eta_values, 75)),
            float(np.percentile(eta_values, 75) / 60),
            float(skew(eta_values)),
            float(kurtosis(eta_values))
        ]
    }
    
    # Category distribution
    recommended = results['recommended_thresholds']
    category_dist = results['category_distribution']
    
    category_summary = {
        'category': ['short', 'medium', 'long'],
        'threshold_seconds': [
            recommended['short_threshold_seconds'],
            f"{recommended['short_threshold_seconds']} - {recommended['long_threshold_seconds']}",
            recommended['long_threshold_seconds']
        ],
        'threshold_minutes': [
            recommended['short_threshold_minutes'],
            f"{recommended['short_threshold_minutes']:.1f} - {recommended['long_threshold_minutes']:.1f}",
            recommended['long_threshold_minutes']
        ],
        'count': [
            category_dist['short']['count'],
            category_dist['medium']['count'],
            category_dist['long']['count']
        ],
        'percentage': [
            category_dist['short']['percentage'],
            category_dist['medium']['percentage'],
            category_dist['long']['percentage']
        ],
        'mean_eta_seconds': [
            category_dist['short']['mean_seconds'],
            category_dist['medium']['mean_seconds'],
            category_dist['long']['mean_seconds']
        ],
        'mean_eta_minutes': [
            category_dist['short']['mean_seconds'] / 60,
            category_dist['medium']['mean_seconds'] / 60,
            category_dist['long']['mean_seconds'] / 60
        ],
        'median_eta_seconds': [
            category_dist['short']['median_seconds'],
            category_dist['medium']['median_seconds'],
            category_dist['long']['median_seconds']
        ],
        'median_eta_minutes': [
            category_dist['short']['median_seconds'] / 60,
            category_dist['medium']['median_seconds'] / 60,
            category_dist['long']['median_seconds'] / 60
        ]
    }
    
    # All method comparisons
    all_methods_data = []
    for method_name, method_data in results['all_methods'].items():
        dist = method_data['distribution']
        all_methods_data.append({
            'method': method_name,
            'short_threshold_seconds': method_data['short_threshold_seconds'],
            'long_threshold_seconds': method_data['long_threshold_seconds'],
            'short_threshold_minutes': method_data['short_threshold_seconds'] / 60,
            'long_threshold_minutes': method_data['long_threshold_seconds'] / 60,
            'short_count': dist['short']['count'],
            'short_percentage': dist['short']['percentage'],
            'medium_count': dist['medium']['count'],
            'medium_percentage': dist['medium']['percentage'],
            'long_count': dist['long']['count'],
            'long_percentage': dist['long']['percentage']
        })
    
    # Save summary statistics
    stats_df = pd.DataFrame(stats_summary)
    stats_path = os.path.join(export_folder, 'eta_analysis_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    
    # Save category summary
    category_df = pd.DataFrame(category_summary)
    category_path = os.path.join(export_folder, 'eta_analysis_categories.csv')
    category_df.to_csv(category_path, index=False)
    
    # Save all methods comparison
    methods_df = pd.DataFrame(all_methods_data)
    methods_path = os.path.join(export_folder, 'eta_analysis_methods.csv')
    methods_df.to_csv(methods_path, index=False)
    
    print(f"✅ Summary CSV files created:")
    print(f"   📈 Statistics: {stats_path}")
    print(f"   🎯 Categories: {category_path}")
    print(f"   🔍 Methods: {methods_path}")

def analyze_edge_route_counts(snapshots_folder, export_folder="./eda_exports", plot_histogram=True):

    os.makedirs(export_folder, exist_ok=True)
    all_files = get_snapshot_files(snapshots_folder)
    if not all_files:
        print("No snapshot files found.")
        return
    
    print(f"⚠️  Large dataset detected ({len(all_files):,} files). Using memory-efficient processing...")

    per_snapshot_counts = []
    per_snapshot_logs = []
    edge_ids = set()

    # Process in chunks to avoid memory issues
    chunk_size = 50  # Process 50 files at a time
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(total_chunks), desc="Processing snapshots", unit="chunk"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_files))
        chunk_files = all_files[start_idx:end_idx]
        
        # Process chunk
        for file in tqdm(chunk_files, desc=f"Chunk {chunk_idx+1}/{total_chunks}", leave=False, unit="file"):
            try:
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
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load {file}: {str(e)}")
                continue

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
    print("🚀 Memory-optimized for large datasets (chunked processing enabled)")

    while True:
        print("\nMain Menu:")
        main_options = [
            "Summarize All Input Features for Preprocessing",
            "Summarize Label Features for Preprocessing",
            "Analyze edge route counts for Preprocessing",
            "Analyze Feature Distribution",
            "Analyze ETA Categories for Classification",
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
        elif main_choice == 4:  # Analyze ETA Categories
            print("\n🔍 Analyzing ETA Categories for Classification...")
            if args.labels_folder and os.path.exists(args.labels_folder):
                # Check if we should use memory-efficient mode for large datasets
                import glob
                label_files = glob.glob(os.path.join(args.labels_folder, "labels_*.json"))
                if len(label_files) > 10000:  # If more than 10k files, suggest memory-efficient mode
                    print(f"⚠️  Large dataset detected ({len(label_files):,} label files)")
                    print("💡 Consider using memory-efficient mode for very large datasets")
                    use_memory_efficient = input("Use memory-efficient mode (skip main CSV)? [y/N]: ").lower().startswith('y')
                    analyze_eta_categories(args.labels_folder, create_main_csv=not use_memory_efficient)
                else:
                    analyze_eta_categories(args.labels_folder, create_main_csv=True)
                print("ETA category analysis complete. Check ./eda_exports/")
            else:
                print(f"❌ Labels folder not found: {args.labels_folder}")
                print("Please ensure the labels folder exists and contains labels_*.json files.")
            continue
        
        elif main_choice == 5:
            labels_file = os.path.join(args.snapshots_folder, "labels.json")
            plot_route_distances_over_time(labels_file)
        
        elif main_choice == 6:
            labels_file = os.path.join(args.snapshots_folder, "labels.json")
            plot_travel_durations_over_time(labels_file)
        
        print("\nWould you like to:")
        next_step = print_menu(["Start Over", "Exit"])
        if next_step == 1:
            print("Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
