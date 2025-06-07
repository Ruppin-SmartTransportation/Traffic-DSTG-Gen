import json
import matplotlib.pyplot as plt
import networkx as nx
import os
import math

# === Configuration ===
SNAPSHOT_PATH = '/media/guy/StorageVolume/traffic_data/step_006120.json'
OUTPUT_PATH = "tools"
SCALE = 5.0
NODE_SIZE = 100

with open(SNAPSHOT_PATH, "r") as f:
    data = json.load(f)

nodes = data["nodes"]
edges = data["edges"]

# Build node info lookup
node_dict = {n["id"]: n for n in nodes}
node_type_map = {n["id"]: n["node_type"] for n in nodes}

G = nx.DiGraph()
pos = {}

# Set node positions and labels
for node in nodes:
    nid = node["id"]
    ntype = node["node_type"]
    if ntype == 0:
        x = node["x"] * SCALE
        y = node["y"] * SCALE
    else:
        x = node.get("current_x", 0) * SCALE
        y = node.get("current_y", 0) * SCALE
       
    pos[nid] = (x, y)
    G.add_node(nid, ntype=ntype)

static_edges = []
dynamic_edges = []

# For each edge in the static network, add only if there are no vehicles
for edge in edges:
    src = edge["from"]
    tgt = edge["to"]
    veh_ids = edge.get("vehicles_on_road", [])
    if not veh_ids:
        static_edges.append((src, tgt))
        G.add_edge(src, tgt)
    else:
        vehicles = {}
        for veh_id in veh_ids:
            # collect all vehicles on this edge
            vehicle = node_dict.get(veh_id)
            if vehicle:
                vehicles[veh_id] = vehicle
        # sort vehicles by their current position
        sorted_vehicles = sorted(vehicles.values(), key=lambda v: v.get("current_position", 0))
        # chain the vehicles on the edge
        for i in range(len(sorted_vehicles) - 1):
            src_veh = sorted_vehicles[i]["id"]
            tgt_veh = sorted_vehicles[i + 1]["id"]
            G.add_edge(src_veh, tgt_veh)
            dynamic_edges.append((src_veh, tgt_veh))
        
        # connect the edge to the first and last vehicle
        if sorted_vehicles: 
            first_veh = sorted_vehicles[0]["id"]
            last_veh = sorted_vehicles[-1]["id"]
            G.add_edge(src, first_veh)
            dynamic_edges.append((src, first_veh))
            G.add_edge(last_veh, tgt)
            dynamic_edges.append((last_veh, tgt))

print(f"Static edges (no vehicles): {len(static_edges)}")
print(f"Dynamic edges (with vehicles): {len(dynamic_edges)}")

plt.figure(figsize=(24, 16))

# Static (no vehicles) edges: grey
nx.draw_networkx_edges(
    G, pos,
    edgelist=static_edges,
    edge_color="grey",
    width=1.5,
    alpha=0.5,
)
# Dynamic (with vehicles) edges: red
nx.draw_networkx_edges(
    G, pos,
    edgelist=dynamic_edges,
    edge_color="red",
    width=1.5,
    alpha=0.5,
)


# Draw all nodes (same size, colored by type)
junctions = [n for n in G.nodes if node_type_map[n] == 0]
vehicles = [n for n in G.nodes if node_type_map[n] == 1]
nx.draw_networkx_nodes(G, pos, nodelist=junctions, node_color="skyblue", node_size=NODE_SIZE, label="Junction")
nx.draw_networkx_nodes(G, pos, nodelist=vehicles, node_color="orange", node_size=NODE_SIZE, label="Vehicle")

# Draw all labels
# nx.draw_networkx_labels(G, pos, labels=None)
id_pos = {id:'center' for id in vehicles}
for vid in vehicles:
     # only use number from label
    name = str(vid).split("_")[-1]
    x, y = pos[vid]
    direction_x = 0
    direction_y = 1
   
    # find the distance to the 2 closest vehicles
    min_distance_1 = float('inf')
    min_distance_2 = float('inf')
    min_id_1 = None
    min_id_2 = None
    for other_vid in vehicles:
        if other_vid != vid:
            other_x, other_y = pos[other_vid]
            distance = math.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
            if distance < min_distance_1:
                min_distance_2 = min_distance_1
                min_id_2 = min_id_1
                min_distance_1 = distance
                min_id_1 = other_vid
    if min_distance_1 < 1000:
        print(f"Distance of {vid } to {min_id_1}: {min_distance_1}")
        print(f"Distance of {vid } to {min_id_2}: {min_distance_2}")
        # set the position of the label differnt than the other 2 closest vehicles
        possible_positions = ['top', 'bottom', 'center', 'baseline', 'center_baseline']
        print(f"Possible positions: {possible_positions}")
        print(f"remove {id_pos[min_id_1]} and {id_pos.get(min_id_2, 'center')}")
        if id_pos[min_id_1] in possible_positions:
            possible_positions.remove(id_pos[min_id_1])
        if min_id_2 is not None and id_pos.get(min_id_2, 'center') in possible_positions:
            possible_positions.remove(id_pos.get(min_id_2, 'center'))
        
        print(f"Possible positions after removal: {possible_positions}")

        id_pos[vid] = possible_positions[0]
        if id_pos[vid] == 'top':
            direction_y = 100

        elif id_pos[vid] == 'bottom':
            direction_y = -100
        elif id_pos[vid] == 'center_baseline':
            direction_x = 100 
    
    plt.text(x+5*direction_x, y + 8*direction_y, name, fontsize=5, color="darkblue", ha='center', va=id_pos[vid])
   

plt.legend(scatterpoints=1, loc='upper right')
plt.title("Traffic Graph: Only Junctions, Vehicles, and Static Edges (no vehicles present on edge)")
plt.axis("off")
plt.tight_layout()

file_name = os.path.basename(SNAPSHOT_PATH).replace(".json", "")
file_name = f"{file_name}_graph.png"
OUTPUT_PATH = os.path.join(OUTPUT_PATH, file_name)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()
