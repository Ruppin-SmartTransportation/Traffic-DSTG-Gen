from sumolib import net
from graph.entities import SimManager
import json
import traci
import traci.constants as tc
from tqdm import tqdm
import os


sumo_binary = "sumo-gui"

# Paths
net_path = "simulation/urban_three_zones.net.xml"
sumo_cfg_path = "simulation/urban_three_zones.sumocfg"
config_path = "simulation.config.json"

if __name__ == "__main__":
    # Load SUMO network
    with open(config_path) as f:
        config = json.load(f)
    net = net.readNet(net_path)
    snapshot_dir = config.get("snapshot_dir", "traffic_data")
    labels_file = os.path.join(snapshot_dir, "labels.json")
    snapshot_interval = config.get("snapshot_interval_sec", 60)
    # Initialize and load simulation
    sim = SimManager(net)
    sim.load_zones()
    sim.populate_vehicles_from_config(config)

    # Summary output
   
    sim.schedule_from_config(config)

    # Launch SUMO with TraCI
    sumo_cmd = [sumo_binary, "-c", sumo_cfg_path, "--start"]
    limit = sim.calculate_simulation_limit(config)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    step = 0
    traci.start(sumo_cmd)
    for step in tqdm(range(limit), desc="Simulation Steps", unit="step"):
        traci.simulationStep()
        sim.update(step, traci)
        sim.dispatch(step, traci)
        if step % snapshot_interval == 0:
            sim.save_snapshot(snapshot_dir, step)

    sim.save_labels_file(labels_file)      
    traci.close()

