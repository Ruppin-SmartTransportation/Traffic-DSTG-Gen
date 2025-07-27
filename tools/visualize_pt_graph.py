import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import argparse
from torch_geometric.data import Data

def load_pt_file(pt_path):
    """Load a PyTorch Geometric data object from .pt file."""
    try:
        data = torch.load(pt_path, map_location='cpu')
        print(f"✅ Loaded PT file: {pt_path}")
        print(f"   Nodes: {data.x.shape[0]}")
        print(f"   Node features: {data.x.shape[1]}")
        print(f"   Edges: {data.edge_index.shape[1]}")
        print(f"   Vehicles: {len(data.vehicle_ids)}")
        return data
    except Exception as e:
        print(f"❌ Error loading PT file: {e}")
        return None

def extract_node_positions(data):
    """Extract node positions from the PT data."""
    # Node features: [node_type, veh_type_oh, speed, acceleration, sin_hour, cos_hour, 
    # sin_day, cos_day, route_length, progress, zone_oh, current_x, current_y, 
    # destination_x, destination_y, current_edge_num_lanes_oh, current_edge_demand, 
    # current_edge_occupancy, j_type]
    
    # Extract positions (indices 16, 17 for current_x, current_y)
    positions = {}
    node_types = {}
    
    for i in range(data.x.shape[0]):
        node_features = data.x[i]
        node_type = int(node_features[0])  # 0 = junction, 1 = vehicle
        
        if node_type == 0:  # Junction
            # Junctions use static positions (indices 16, 17)
            x = float(node_features[16])
            y = float(node_features[17])
            node_id = data.junction_ids[i] if i < len(data.junction_ids) else f"J{i}"
        else:  # Vehicle
            # Vehicles use current positions (indices 16, 17)
            x = float(node_features[16])
            y = float(node_features[17])
            vehicle_idx = i - len(data.junction_ids)
            node_id = data.vehicle_ids[vehicle_idx] if vehicle_idx < len(data.vehicle_ids) else f"V{i}"
        
        positions[node_id] = (x, y)
        node_types[node_id] = node_type
    
    return positions, node_types

def build_networkx_graph(data, positions, node_types):
    """Build NetworkX graph from PT data."""
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, (x, y) in positions.items():
        G.add_node(node_id, pos=(x, y), type=node_types[node_id])
    
    # Add edges
    edge_index = data.edge_index.numpy()
    edge_types = data.edge_type.numpy() if hasattr(data, 'edge_type') else None
    
    static_edges = []
    dynamic_edges = []
    
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        tgt_idx = edge_index[1, i]
        
        # Map indices to node IDs
        if src_idx < len(data.junction_ids):
            src_id = data.junction_ids[src_idx]
        else:
            vehicle_idx = src_idx - len(data.junction_ids)
            src_id = data.vehicle_ids[vehicle_idx]
            
        if tgt_idx < len(data.junction_ids):
            tgt_id = data.junction_ids[tgt_idx]
        else:
            vehicle_idx = tgt_idx - len(data.junction_ids)
            tgt_id = data.vehicle_ids[vehicle_idx]
        
        # Determine edge type
        edge_type = edge_types[i] if edge_types is not None else 0
        
        if edge_type == 0:  # Static edge
            static_edges.append((src_id, tgt_id))
        else:  # Dynamic edge (vehicle-related)
            dynamic_edges.append((src_id, tgt_id))
        
        G.add_edge(src_id, tgt_id, type=edge_type)
    
    return G, static_edges, dynamic_edges

def visualize_graph(G, positions, node_types, static_edges, dynamic_edges, output_path, title="Traffic Graph from PT File"):
    """Create and save the graph visualization."""
    plt.figure(figsize=(20, 16))
    
    # Extract positions for NetworkX
    pos = {node: positions[node] for node in G.nodes()}
    
    # Separate nodes by type
    junctions = [node for node in G.nodes() if node_types[node] == 0]
    vehicles = [node for node in G.nodes() if node_types[node] == 1]
    
    # Draw edges
    if static_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=static_edges,
            edge_color="grey",
            width=1.0,
            alpha=0.6,
            label="Static edges"
        )
    
    if dynamic_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=dynamic_edges,
            edge_color="red",
            width=1.5,
            alpha=0.8,
            label="Dynamic edges"
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=junctions, 
        node_color="skyblue", 
        node_size=150, 
        label="Junctions"
    )
    
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=vehicles, 
        node_color="orange", 
        node_size=100, 
        label="Vehicles"
    )
    
    # Add vehicle labels (simplified)
    vehicle_labels = {}
    for vehicle_id in vehicles:
        # Extract vehicle number from ID
        if vehicle_id.startswith('veh_'):
            vehicle_num = vehicle_id.split('_')[1]
        else:
            vehicle_num = vehicle_id
        vehicle_labels[vehicle_id] = vehicle_num
    
    # Draw labels with reduced overlap
    nx.draw_networkx_labels(
        G, pos,
        labels=vehicle_labels,
        font_size=6,
        font_color="darkblue",
        font_weight="bold"
    )
    
    plt.legend()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize traffic graph from PT file")
    parser.add_argument('--pt_file', type=str, required=True, help='Path to the .pt file')
    parser.add_argument('--output', type=str, default='visualization.png', help='Output image path')
    parser.add_argument('--title', type=str, default='Traffic Graph from PT File', help='Plot title')
    parser.add_argument('--no_show', action='store_true', help='Do not display the plot (save only)')
    
    args = parser.parse_args()
    
    # Load PT file
    data = load_pt_file(args.pt_file)
    if data is None:
        return
    
    # Extract node positions and types
    positions, node_types = extract_node_positions(data)
    
    # Build NetworkX graph
    G, static_edges, dynamic_edges = build_networkx_graph(data, positions, node_types)
    
    print(f"📊 Graph Statistics:")
    print(f"   Total nodes: {G.number_of_nodes()}")
    print(f"   Junctions: {len([n for n in G.nodes() if node_types[n] == 0])}")
    print(f"   Vehicles: {len([n for n in G.nodes() if node_types[n] == 1])}")
    print(f"   Total edges: {G.number_of_edges()}")
    print(f"   Static edges: {len(static_edges)}")
    print(f"   Dynamic edges: {len(dynamic_edges)}")
    
    # Create visualization
    visualize_graph(G, positions, node_types, static_edges, dynamic_edges, args.output, args.title)
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main() 