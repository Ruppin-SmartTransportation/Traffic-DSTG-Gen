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
│
├── main.py                       # Main entry point for running graph generation
├── requirements.txt              # All dependencies with pinned versions
├── simulation.config.json        # Config for running scenarios
├── graph/                        # Core graph and entity logic
│   ├── __init__.py
│   └── entities.py
├── simulation/                   # SUMO scenarios, networks, and simulation data
│   ├── urban_three_zones.net.xml     # Three-zone network definition
│   ├── urban_three_zones.sumocfg     # SUMO config file
│   ├── vehicle_types.add.xml
│   ├── view_settings.xml
│   └── ... (other scenario files)
├── tools/                        # Data analysis and conversion utilities
│   ├── EDA.py                    # Exploratory Data Analysis
│   └── convert_to_pt.py          # Convert to PyTorch Geometric format
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

### Convert simulation output to PyTorch Geometric format:

```bash
python tools/convert_to_pt.py --input <simulation_output_dir> --output <dataset.pt>
```

---

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

### Example Usage: Loading a `.pt` Graph

```python
import torch
data = torch.load('traffic_data/step_600.pt')
print(data)
# Data(x=[n_nodes, n_features], edge_index=[2, n_edges], edge_attr=[n_edges, n_edge_features], ...)
```

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