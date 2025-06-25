import torch
import pandas as pd
import argparse
import os

def analyze_snapshot(path):
    data = torch.load(path)
    results = []

    for attr in dir(data):
        if attr.startswith("_") or attr in {"edge_index", "batch"}:
            continue
        value = getattr(data, attr, None)

        if isinstance(value, torch.Tensor):
            if value.dim() == 2:
                for i in range(value.size(1)):
                    col = value[:, i]
                    dtype = str(col.dtype)
                    stats = {
                        "feature_group": attr,
                        "feature_index": i,
                        "dtype": dtype,
                        "count": col.numel()
                    }
                    if col.is_floating_point():
                        stats["mean"] = float(col.mean().item())
                        stats["std"] = float(col.std().item())
                    else:
                        stats["mean"] = float(col.float().mean().item())
                        stats["std"] = float(col.float().std().item())
                    results.append(stats)
            else:
                dtype = str(value.dtype)
                stats = {
                    "feature_group": attr,
                    "feature_index": None,
                    "dtype": dtype,
                    "count": value.numel()
                }
                if value.is_floating_point():
                    stats["mean"] = float(value.mean().item())
                    stats["std"] = float(value.std().item())
                else:
                    stats["mean"] = float(value.float().mean().item())
                    stats["std"] = float(value.float().std().item())
                results.append(stats)

    df = pd.DataFrame(results)
    print(df)
    eda_folder = "eda_exports"
    file_name = "snapshot_feature_stats.csv"
    file_path = os.path.join(eda_folder, file_name)
    df.to_csv(file_path, index=False)
    print(f"✅ Saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="/home/guy/Projects/Traffic/traffic_data_pt/step_2231640.pt",
        help="Folder with .pt graph files"
    )
    args = parser.parse_args()
    analyze_snapshot(args.path)
