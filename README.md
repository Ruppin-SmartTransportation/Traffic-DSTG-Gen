# Traffic-DSTG-Gen

**Dynamic Spatio-Temporal Graph Generator for Traffic Simulation Data**

![License](https://img.shields.io/github/license/turgibot/Traffic-DSTG-Gen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![SUMO](https://img.shields.io/badge/SUMO-1.22.0-green?style=flat-square)

---

**Traffic-DSTG-Gen** is an open-source tool for generating dynamic, spatio-temporal graph datasets from urban traffic simulations.  
It is designed for research and development in traffic forecasting, intelligent transportation systems, and machine learning—especially graph neural networks (GNNs).

This toolkit integrates with SUMO (Simulation of Urban Mobility) and supports user-defined scenarios, including complex urban environments with multiple traffic zones.

---

## Demo

> **Simulation of urban_three_zones scenario**  

![Simulation Demo](tools/demo.gif)

---

## Features

- **Dynamic Graph Generation**  
  Converts traffic simulation data into dynamic, spatio-temporal graphs with both vehicles and junctions as nodes.

- **Three-Zone Urban Network**  
  Includes a configurable SUMO scenario (`urban_three_zones.net.xml`) with residential, commercial, and attraction zones, simulating realistic city dynamics.

- **PyTorch Geometric Compatibility**  
  Exports datasets ready for modern GNN frameworks.

- **Scenario Customization**  
  Easily modify simulation configuration and map layouts for custom experiments.

- **Visualization Tools**  
  Built-in tools for exploratory data analysis and graph visualization.

---



## Novelty: Dynamic Graph Construction with Vehicle Nodes

A key innovation of **Traffic-DSTG-Gen** is the representation of **dynamic traffic states as graphs where both junctions *and* vehicles are nodes**. Unlike traditional approaches—which typically only model junctions and road segments—our framework dynamically rewires the graph at each simulation step to reflect the actual positions of vehicles on the road network.

- **Junctions** (blue nodes): Represent intersections in the traffic network.
- **Vehicles** (orange nodes): Represent every vehicle currently active in the network.
- **Static edges** (grey): Represent roads that currently have no vehicles on them.
- **Dynamic edges** (red): For every road segment with vehicles, the edge is replaced by a *chain* of red edges connecting:
    - The source junction to the first vehicle on the edge,
    - Consecutive vehicles along the edge (ordered by position),
    - The last vehicle to the destination junction.

This graph snapshot visually demonstrates how the traffic graph changes with the real-time flow of vehicles, making it ideally suited for advanced **spatio-temporal graph neural network (STGNN) research**.

![Dynamic Traffic Graph Example](tools/step_006120_graph.png)

*Above: Example snapshot. Orange nodes = vehicles, blue nodes = junctions. Grey lines = empty roads; red chains = vehicle "convoys" on busy edges. Node labels indicate IDs.*


## Simulation Scenario: Three Urban Zones

The provided `urban_three_zones.net.xml` network models a realistic city environment with:

- **Zone A:** Residential area  
- **Zone B:** Commercial center  
- **Zone C:** Residential zone with major attractions (weekend/evening hotspots)

This scenario supports configurable rush-hour flows, attraction-based traffic, and complex real-world patterns—ideal for evaluating spatio-temporal GNNs and traffic prediction algorithms.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ruppin-SmartTransportation/Traffic-DSTG-Gen.git
cd Traffic-DSTG-Gen
```

### 2. Set up the environment

It is **strongly recommended** to use a Python virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
```

### 3. Install dependencies

All dependencies (with pinned versions for reproducibility) are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

**Required versions:**
```
matplotlib==3.7.5
pandas==2.0.3
sumolib==1.22.0
torch==2.4.1+cu121
torch_geometric==2.6.1
tqdm==4.67.1
traci==1.22.0
```

> **Note:**  
> - Ensure [SUMO](https://sumo.dlr.de/docs/Downloads.php) (version 1.22.0) is installed and available on your system PATH.
> - CUDA-enabled PyTorch (`torch==2.4.1+cu121`) is recommended for GPU acceleration; adapt the version if needed for your system.

---

## Project Structure

```
Traffic-DSTG-Gen/
├── main.py # Entrypoint: simulation control & graph generation
├── simulation.config.json # Global config for zones, vehicles, intervals

├── graph/
│ ├── init.py
│ └── entities.py # Classes for graph node/edge abstractions

├── simulation/
│ ├── init.py
│ ├── *.txt # Zone-specific traffic sources (e.g., park1, stadium1)
│ ├── *.net.xml, *.sumocfg # SUMO network & simulation configs
│ ├── vehicle_types.add.xml # Vehicle type definitions
│ └── view_settings.xml # SUMO GUI visual config

├── tools/
│ ├── convert_to_pt.py # Converts graph snapshots to PyG .pt format
│ ├── create_labels_json.py # Label creation for supervised learning
│ ├── EDA.py # Exploratory data analysis
│ ├── pt_validation.py # Checks .pt output structure
│ ├── visualize_graph.py # Graph snapshot visualization (e.g., NetworkX)
│ ├── labels_.json # Label data (sampled)
│ ├── step_.json/.png # Sample snapshot data and graph images
│ └── demo.gif # Animated simulation demo

├── eda_exports/ # EDA outputs (generated during analysis)
├── logs/ # Logs from SUMO or run-time
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

---

## Example Usage

### Generate a dynamic graph dataset from the three-zone simulation:

```bash
python main.py --config simulation/simulation.config.json --net simulation/urban_three_zones.net.xml
```

- Edit `simulation.config.json` to configure simulation length, traffic flow, and other parameters.
- The output dataset (adjacency matrices, node/edge features, etc.) will be saved as specified in your configuration.
  
## Sample Output

### Example Snapshot Output (`.json`)

```json
{
  "step": 600,
  "nodes": [
    {
      "id": "veh_23",
      "node_type": 1,
      "speed": 12.3,
      "current_zone": "B",
      "origin_zone": "A",
      "destination_zone": "C",
      "...": "..."
    },
    {
      "id": "junction_4",
      "node_type": 0,
      "zone": "B",
      "type": "traffic_light",
      "...": "..."
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "from": "veh_23",
      "to": "junction_4",
      "density": 0.3,
      "zone": "B",
      "...": "..."
    }
  ]
}
```
*Fields truncated for clarity—see real output for full details.*
## 🧪 Exploratory Data Analysis (EDA)

The repository includes a powerful CLI-based EDA tool: `tools/EDA.py`, designed for:

- 📊 **Analyzing feature distributions** across vehicles, junctions, and edges  
- 🚨 **Detecting outliers**  
- 🔍 **Summarizing features** for preprocessing  
- 📤 **Exporting plots and stats** into `eda_exports/`

### 🔍 Supported Features

You can analyze features such as:
- `speed`, `zone`, `density`, `origin_zone`, `destination_zone`, etc.
- Works for any entity type: **vehicles**, **junctions**, or **edges**

### 🖥️ Example: Run the Toolkit

```bash
python tools/EDA.py --snapshots_folder traffic_data
```

You’ll be prompted to:
1. Choose entity type (e.g., vehicles)
2. Select a feature (e.g., speed)
3. Choose analysis type:
   - Histogram
   - Boxplot
   - Stats (mean, std, skew, kurtosis)
   - Outliers
   - Skewness (KDE)
   - Normalization preview

### 🗃️ Exported Results

Saved in `./eda_exports/`, including:
- `*_histogram.png` — feature distribution
- `*_boxplot.png` — outlier sensitivity
- `*_stats.txt` — statistical summary
- `*_outliers.txt` — detected anomalies
- `*_normalization_preview.png` — min-max & z-score comparison
- `*_feature_summary.csv` — summary across all features (for preprocessing)

---

This tool is especially useful for:
- Selecting robust and informative features for your STGNN models
- Understanding scaling requirements and outlier behavior
- Detecting data imbalance or skewed inputs
- Creating thesis-ready EDA summaries and plots

### Convert simulation output to PyTorch Geometric format:

```bash
python tools/convert_to_pt.py --input <simulation_output_dir> --output <dataset.pt>
```

---


# 🧰 Dataset Preparation Workflow

This project includes four key scripts to transform SUMO simulation output into clean, GNN-ready `.pt` graph files. These tools support a full preprocessing pipeline, from raw snapshots to validated PyTorch Geometric datasets.

---

## 🔁 Full Workflow Overview

1. **[Simulation]** Generate traffic snapshots and a global `labels.json` from SUMO using `main.py`
2. **[Labeling]** Generate per-snapshot ETA labels using `create_labels_json.py`
3. **[EDA]** Generate feature statistics using `EDA.py`
4. **[Conversion]** Convert data to `.pt` format using `convert_to_pt.py`
5. **[Validation]** Verify `.pt` integrity using `pt_validation.py`

---

## 1. [main.py] – Generate Snapshots & Ground Truth Labels

This script runs the SUMO simulation and generates:
- Snapshot files: `step_XXXX.json`
- Global ground truth label file: `labels.json`

### ✅ Example:
```bash
python main.py --config simulation/simulation.config.json --sumo-gui
```

Outputs are written to `traffic_data/`

---

## 2. `create_labels_json.py` – Generate Per-Snapshot Ground Truth Labels

This script creates per-snapshot label files (`labels_*.json`) based on the global `labels.json`.

### ✅ Command:
```bash
python tools/create_labels_json.py \
  --snapshots_folder traffic_data \
  --gt_labels_path traffic_data/labels.json \
  --output_labels_folder traffic_data/labels
```

---

## 3. `EDA.py` – Generate Feature Summaries

This script analyzes simulation data and exports feature statistics needed for preprocessing and normalization.

### ✅ Command:
```bash
python tools/EDA.py --snapshots_folder traffic_data
```

Outputs are saved in `eda_exports/`:
- `vehicle_feature_summary.csv`
- `junction_feature_summary.csv`
- `edge_feature_summary.csv`

---

## 4. `convert_to_pt.py` – Convert to PyTorch Geometric Format

Transforms snapshot+label `.json` files into `.pt` graph datasets.

### ✅ Command:
```bash
python tools/convert_to_pt.py \
  --snapshots_folder traffic_data \
  --labels_folder traffic_data/labels \
  --eda_folder eda_exports \
  --out_graph_folder traffic_data_pt
```

---

## 5. `pt_validation.py` – Validate .pt Graph Files

Randomly samples `.pt` files and compares fields against raw `.json` and labels.

### ✅ Command:
```bash
python tools/pt_validation.py \
  --pt_folder traffic_data_pt \
  --gt_folder traffic_data/labels \
  --snapshot_folder traffic_data \
  --eda_folder eda_exports \
  --n_samples 10
```

---

## 📦 Summary Table

| Step | Script                  | Input(s)                                 | Output(s)                               |
|------|--------------------------|-------------------------------------------|------------------------------------------|
| 1    | `main.py`                | SUMO config, network files                | `step_*.json`, `labels.json`             |
| 2    | `create_labels_json.py` | `step_*.json`, `labels.json`              | `labels_*.json`                          |
| 3    | `EDA.py`                 | `step_*.json`                             | `vehicle/junction/edge_feature_summary.csv` |
| 4    | `convert_to_pt.py`       | `step_*.json`, `labels_*.json`, `.csv`    | `.pt` graph files (PyG format)           |
| 5    | `pt_validation.py`       | `.pt`, `step_*.json`, `labels_*.json`, `.csv` | Validation logs (stdout)              |
|------|--------------------------|-------------------------------------------|------------------------------------------|

---

### Example Usage: Loading a `.pt` Graph

```python
import torch
data = torch.load('traffic_data/step_600.pt')
print(data)
# Data(x=[n_nodes, n_features], edge_index=[2, n_edges], edge_attr=[n_edges, n_edge_features], ...)
```

---


## Contributing

Contributions are welcome!  
Feel free to open issues, submit pull requests, or suggest improvements for the codebase or documentation.

---

## Citation

If you use Traffic-DSTG-Gen in your research or publication, please cite this repository:

```bibtex
@misc{trafficdstggen2024,
  author = {Your Name and Collaborators},
  title = {Traffic-DSTG-Gen: Dynamic Spatio-Temporal Graph Generator for Traffic Simulation Data},
  year = {2024},
  howpublished = {\url{https://github.com/Ruppin-SmartTransportation/Traffic-DSTG-Gen}},
}
```

---

## Acknowledgments

This project was funded by the **Ministry of Innovation, Science, and Technology** (MOST) and the **Ministry of Transport and Road Safety, Israel** (2024–2027) as part of the national program for Smart Transportation research (Grant #0007846).  
Special thanks to the Division of Planning & Development, Ruppin Academic Center, and project coordinator **Dr. Nadav Voloch**.

This research is based on, and extends, the methodology presented in:  
**Voloch, N., & Voloch-Bloch, N. (2021). "Finding the fastest navigation route by real-time future traffic estimations." 2021 IEEE International Conference on Microwaves, Antennas, Communications and Electronic Systems (COMCAS), pp. 13-16. IEEE.**  
Available at: [https://www.researchgate.net/publication/356828106_Finding_the_fastest_navigation_rout_by_real-time_future_traffic_estimations](https://www.researchgate.net/publication/356828106_Finding_the_fastest_navigation_rout_by_real-time_future_traffic_estimations)

We gratefully acknowledge the support and comments of our collaborators and reviewers, and the open-source SUMO and PyTorch Geometric communities.

> **Note:** All publications arising from this work must acknowledge the support of the Ministry of Innovation, Science, and Technology and the Ministry of Transport and Road Safety, as per the grant requirements.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

*For questions or collaboration inquiries, please contact [turgibot@gmail.com].*