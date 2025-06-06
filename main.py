import argparse
from sumolib import net as sumo_net
from graph.entities import SimManager
import json
import traci
import traci.constants as tc
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Traffic-DSTG-Gen: Generate dynamic spatio-temporal graph datasets from SUMO traffic simulation.")
    parser.add_argument('--config', default="simulation.config.json", help="Path to simulation config JSON file.")
    parser.add_argument('--net', default="simulation/urban_three_zones.net.xml", help="Path to SUMO network file (.net.xml).")
    parser.add_argument('--sumocfg', default="simulation/urban_three_zones.sumocfg", help="Path to SUMO config file (.sumocfg).")
    parser.add_argument('--sumo-gui', action='store_true', help="Use SUMO GUI (default: False, use CLI)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sumo_binary = "sumo-gui" if args.sumo_gui else "sumo"

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Load SUMO network
    net = sumo_net.readNet(args.net)
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
    sumo_cmd = [sumo_binary, "-c", args.sumocfg, "--start"]
    limit = sim.calculate_simulation_limit(config)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)

    traci.start(sumo_cmd)
    for step in tqdm(range(limit), desc="Simulation Steps", unit="step"):
        traci.simulationStep()
        sim.update(step, traci)
        sim.dispatch(step, traci)
        if step % snapshot_interval == 0:
            sim.save_snapshot(snapshot_dir, step)

    sim.save_labels_file(labels_file)
    traci.close()
