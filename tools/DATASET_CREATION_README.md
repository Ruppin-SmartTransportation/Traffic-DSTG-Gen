# Traffic Dataset Creation Guide

This document explains the process and structure for creating graph-based datasets from traffic simulation snapshots, as implemented in `tools/dataset_creator.py`.

---

## Node Features (Junctions & Vehicles)

| Index | Feature Name        | Notes                                                   |
| ----- | ------------------- | ------------------------------------------------------- |
| 0     | `node_type`         | 0 = junction, 1 = vehicle                               |
| 1-3   | `veh_type_oh`       | ['bus', 'passenger', 'truck'] one-hot; [0,0,0] for junctions |
| 4     | `speed`             | min-max normalized if normalize, else raw               |
| 5     | `acceleration`      | min-max normalized if normalize, else raw               |
| 6     | `sin_hour`          | Time of day (sin)                                       |
| 7     | `cos_hour`          | Time of day (cos)                                       |
| 8     | `sin_day`           | Day of week (sin)                                       |
| 9     | `cos_day`           | Day of week (cos)                                       |
| 10    | `route_length`      | min-max normalized if normalize, else raw               |
| 11    | `route_length_left` | min-max normalized if normalize, else raw               |
| 12-15 | `zone_oh`           | One-hot of zone (4 zones = 4 dims)                      |
| 16    | `current_x`         | min-max normalized if normalize, else raw               |
| 17    | `current_y`         | min-max normalized if normalize, else raw               |
| 18    | `j_type`            | Junction type (priority/traffic_light); 0 for vehicles  |

Total: **19 features per node**

---

## Edge Features (Static Edges)

| Index | Feature Name        | Notes                                                   |
| ----- | ---------------------- | ---------------------------------------------------- |
| 0     | `avg_speed`            | min-max normalized if normalize, else raw            |
| 1-3   | `num_lanes`            | One-hot of 1-3 lanes                                 |
| 4     | `length`               | min-max normalized if normalize, else raw            |
| 5     | `edge_demand`          | log+z-score normalized if log_normalize, else min-max if normalize, else raw |
| 6     | `edge_occupancy`       | log+z-score normalized if log_normalize, else min-max if normalize, else raw |

Total: **7 features per edge**

---

## Normalization Options

- If `normalize` is True, min-max normalization is applied to most continuous features: `(value - min) / (max - min)`.
- If `log_normalize` is True, log+z-score normalization is applied to skewed/count features (`edge_demand`, `edge_occupancy`): `log1p(value) - mean / std`.
- If both are False, raw values are used.
- For `edge_demand` and `edge_occupancy`, `log_normalize` takes precedence over `normalize`.

---

## Indexing

- **Junctions:** `junctions_id_to_idx = {'J1': 0, 'J2': 1, ...}`
- **Vehicles:** `vehicles_id_to_idx = {vid: idx + offset for idx, vid in enumerate(curr_vehicle_ids)}`
- **Nodes:** `nodes_id_to_idx = {**junction_id_to_idx, **vehicle_id_to_idx}`
- **Edges:** `edges_id_to_idx = {'AX3AX2': 0, 'AX4AX3': 1, ...}` (only static edges)

---

## Edge Index and Types

- `edge_index = [[src_node_0, ...], [tgt_node_0, ...]]`
- `edge_type = [0, 1, 1, 2, ...]`

| Edge Type Code | Description        |
| -------------- | ------------------ |
| 0              | Static road edge   |
| 1              | Junction → Vehicle |
| 2              | Vehicle → Junction |
| 3              | Vehicle → Vehicle  |

Static edges (type 0) are always listed first. Dynamic edges (types 1, 2, 3) are added per snapshot to represent current vehicle positions.

---

## PyTorch Geometric Data Object Structure

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

---

## Label Filtering
- Only vehicles with `180 <= duration <= 99th percentile` are included.

---

## Dynamic Edges
- Dynamic edges are constructed per snapshot to represent the current traffic flow (junction→vehicle, vehicle→vehicle, vehicle→junction). Their edge features are filled with zeros.

---

## Notes
- All normalization/statistics are computed from summary CSVs in the `eda_exports` folder.
- The dataset creation script validates data consistency and feature dimensions at every step.

---

For further details, see the code in `tools/dataset_creator.py`. 