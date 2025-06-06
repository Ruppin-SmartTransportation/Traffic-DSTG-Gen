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
git clone https://github.com/yourusername/Traffic-DSTG-Gen.git
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
  howpublished = {\url{https://github.com/yourusername/Traffic-DSTG-Gen}},
}
```

---

## Acknowledgments

This project was supported by [Your Grant/Institute Name].  
Special thanks to [Your Advisor/Collaborators] and the open-source SUMO and PyTorch Geometric communities.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

*For questions or collaboration inquiries, please contact [your.email@institute.edu].*