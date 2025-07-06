import re
import os
import random
import traci.constants as tc
from collections import defaultdict
import logging
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
import json


class Logger:
    def __init__(
        self,
        name='SimManager',
        log_file='logs/traffic_prediction.log',
        level=logging.INFO,
        max_bytes=5*1024*1024,  # 5 MB per file
        backup_count=5          # Keep up to 5 rotated logs
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # Prevent handler duplication
        if not self.logger.hasHandlers():
            # Ensure the log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            # Rotating file handler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Uncomment the following lines to add a console stream handler
            # Stream handler (console)
            # stream_handler = logging.StreamHandler()
            # stream_handler.setFormatter(file_formatter)
            # self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger


class Junction:
    """
    Represents a fixed junction point in the traffic network.
    Holds incoming and outgoing road connections and basic spatial metadata.
    """
    def __init__(self, junction_id, x=0.0, y=0.0, junc_type="priority", zone=None):
        self.id = junction_id
        self.x = x
        self.y = y
        self.type = junc_type
        self.zone = zone
        self.node_type = 0  # 0 for junction, 1 for vehicle

        self.incoming_roads = set()  # Set of incoming road IDs
        self.outgoing_roads = set()  # Set of outgoing road IDs

    def add_incoming(self, road_id):
        self.incoming_roads.add(road_id)

    def add_outgoing(self, road_id):
        self.outgoing_roads.add(road_id)

    def to_dict(self):
        return {
            "id": self.id,
            "node_type": self.node_type,
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "zone": self.zone,
            "incoming": sorted(self.incoming_roads),
            "outgoing": sorted(self.outgoing_roads)
        }


class Road:
    """
    Represents a road (edge) connecting two junctions.
    Includes static properties such as speed, length, and lane count.
    """
    def __init__(self, road_id, from_junction, to_junction, speed=13.89, length=100.0, num_lanes=1, zone=None):
        self.id = road_id
        self.from_junction = from_junction
        self.to_junction = to_junction
        self.speed = speed
        self.length = length
        self.num_lanes = num_lanes
        self.zone = zone  # Zone label (e.g. 'A', 'B', 'C', or 'H')
        self.vehicles_on_road = {}
        self.density = 0.0  # Computed as vehicles / (length * num_lanes)
        self.avg_speed = 0.0


    def set_density(self):
        if self.length > 0 and self.num_lanes > 0:
            self.density = len(self.vehicles_on_road) / (self.length * self.num_lanes)
        else:
            self.density = 0.0

    def add_vehicle_and_update(self, vehicle):
        """
        Adds a vehicle ID to the road and updates density.
        """
        self.vehicles_on_road[vehicle.id] = vehicle.speed
        self.set_density()
        self.update_avg_speed()

    def remove_vehicle_and_update(self, vehicle):
        """
        Removes a vehicle ID from the road and updates density.
        """
        del self.vehicles_on_road[vehicle.id]
        self.set_density()
        self.update_avg_speed()


    def get_density(self):
        """
        Returns the current density of the road.
        """
        return self.density
    def update_avg_speed(self):
        """
        Updates the average speed of vehicles on this road.
        """
        if not self.vehicles_on_road:
            self.avg_speed = 0.0
            return

        total_speed = sum(self.vehicles_on_road.values())
        self.avg_speed = total_speed / len(self.vehicles_on_road)
        
    

    def to_dict(self):
        return {
            "id": self.id,
            "from": self.from_junction,
            "to": self.to_junction,
            "speed": self.speed,
            "length": self.length,
            "num_lanes": self.num_lanes,
            "zone": self.zone,
            "density": self.density,
            "avg_speed": self.avg_speed,
            "vehicles_on_road": sorted(self.vehicles_on_road.keys())    
        }


class Vehicle:
    """
    Represents a dynamic vehicle in the simulation.
    Tracks position, movement, physical characteristics, and zone associations.
    """
    def __init__(
        self,
        vehicle_id,
        vehicle_type,
        current_edge,
        current_position=0.0,
        speed=0.0,
        acceleration=0.0,
        route=None,
        route_left=None,
        length=4.5,
        width=1.8,
        height=1.5,
        current_x=None,
        current_y=None,
        current_zone=None,
        color='green',
        status="parked",
        is_stagnant=False
    ):
        # static properties
        self.id = vehicle_id
        self.vehicle_type = vehicle_type
        self.width = width
        self.length = length
        self.height = height
        self.color = color
        
        # dynamic properties
        self.speed = speed
        self.acceleration = acceleration
        self.current_edge = current_edge
        self.current_position = current_position
        self.current_x = current_x
        self.current_y = current_y
        self.current_zone = current_zone
        

        self.node_type = 1  # 0 for junction, 1 for vehicle
        self.is_stagnant = is_stagnant  # True if vehicle is not tracked by the model

        # routing and scheduling properties
        self.status = status  # e.g., "moving", "parked"
        self.scheduled = [False, False, False, False]  # True if vehicle is already scheduled for dispatch for the current week
       
        self.route = route if route else []
        self.route_left = route_left if route_left else []
        self.route_length = 0.0
        self.route_length_left = 0.0
       
        self.origin_name = None
        self.origin_zone = None
        self.origin_x = None
        self.origin_y = None
        self.origin_edge = None
        self.origin_position = None
        self.origin_step = None

        self.destinations = {
            "home": {"edge": self.current_edge, "position": self.current_position},
            "work": None,
            "friend1": None,
            "friend2": None,
            "friend3": None,
            "park1": None,
            "park2": None,
            "park3": None,
            "park4": None,
            "stadium1": None,
            "stadium2": None,
            "restaurantA": None,
            "restaurantB": None,
            "restaurantC": None
        }
        self.destination_name = None
        self.destination_zone = None
        self.destination_x = None
        self.destination_y = None
        self.destination_edge = None
        self.destination_position = None
        self.destination_step = None

    def to_dict(self):
        return {
            # Static properties
            "id": self.id,
            "node_type": self.node_type,
            "vehicle_type": self.vehicle_type,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            # Dynamic properties
            "speed": self.speed,
            "acceleration": self.acceleration,
            "current_x": self.current_x,
            "current_y": self.current_y,
            "current_zone": self.current_zone,
            "current_edge": self.current_edge,
            "current_position": self.current_position,

            # Routing and scheduling properties
            
            "origin_name": self.origin_name,
            "origin_zone": self.origin_zone,
            "origin_edge": self.origin_edge,
            "origin_position": self.origin_position,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "origin_start_sec": self.origin_step,

            "route": self.route,
            "route_length": self.route_length,
            "route_left": self.route_left,
            "route_length_left": self.route_length_left,
            
            "destination_name": self.destination_name,
            "destination_edge": self.destination_edge,
            "destination_position": self.destination_position,
            "destination_x": self.destination_x,
            "destination_y": self.destination_y
        }


class Zone:
    """
    Represents a traffic zone (e.g., 'A', 'B', 'C', 'H').
    Tracks all edges and junctions belonging to the zone,
    as well as vehicles that originated or are currently located in the zone.
    """
    def __init__(self, zone_id, description=None):
        self.id = zone_id
        self.description = description  # Optional textual description of the zone
        self.edges = set()
        self.junctions = set()
        self.original_vehicles = set()  # Vehicles that originated here
        self.current_vehicles = set()   # Vehicles currently here

    def add_edge(self, edge_id):
        self.edges.add(edge_id)

    def add_junction(self, junction_id):
        self.junctions.add(junction_id)

    def add_original_vehicle(self, vehicle_id):
        self.original_vehicles.add(vehicle_id)

    def add_current_vehicle(self, vehicle_id):
        self.current_vehicles.add(vehicle_id)

    def remove_current_vehicle(self, vehicle_id):
        self.current_vehicles.discard(vehicle_id)

    def get_random_edge(self):
        import random
        return random.choice(list(self.edges)) if self.edges else None

    def get_random_junction(self):
        import random
        return random.choice(list(self.junctions)) if self.junctions else None

    def get_random_vehicle(self):
        import random
        return random.choice(list(self.original_vehicles)) if self.original_vehicles else None

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "edges": sorted(self.edges),
            "junctions": sorted(self.junctions),
            "original_vehicles": sorted(self.original_vehicles),
            "current_vehicles": sorted(self.current_vehicles)
        }


# Keep these at the bottom so that entity classes are defined first
class DataBase:
    """
    Centralized store for all simulation entities:
    roads, junctions, vehicles, and zones.
    SimManager uses this to read/write state.
    """
    def __init__(self):
        self.roads = {}       # edge_id -> Road
        self.junctions = {}   # junction_id -> Junction
        self.vehicles = {}    # vehicle_id -> Vehicle
        self.zones = {}       # zone_id -> Zone

    def add_road(self, road):
        self.roads[road.id] = road

    def add_junction(self, junction):
        self.junctions[junction.id] = junction

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.id] = vehicle

    def update_junction(self, junction_id, incoming_roads=None, outgoing_roads=None):
        junction = self.get_junction(junction_id)
        if not junction:
            return
        if incoming_roads is not None:
            junction.incoming_roads = set(incoming_roads)
        if outgoing_roads is not None:
            junction.outgoing_roads = set(outgoing_roads)

    def add_zone(self, zone):
        self.zones[zone.id] = zone

    def get_road(self, road_id):
        return self.roads.get(road_id)

    def get_junction(self, junction_id):
        return self.junctions.get(junction_id)

    def get_vehicle(self, vehicle_id):
        return self.vehicles.get(vehicle_id)

    def get_zone(self, zone_id):
        return self.zones.get(zone_id)
    
    def print_zone_statistics(self):
        print("\n--- Simulation Zone Statistics ---")
        for zid, zone in self.zones.items():
            print(f"\nZone {zid}:")

            # Roads by lane count
            lane_counts = {}
            for eid in zone.edges:
                road = self.get_road(eid)
                lane_counts[road.num_lanes] = lane_counts.get(road.num_lanes, 0) + 1
            total_roads = sum(lane_counts.values())
            print(f"  Roads: {total_roads}")
            for lanes, count in sorted(lane_counts.items()):
                print(f"    {count} with {lanes} lane(s)")

            # Junctions and traffic lights
            total_junctions = len(zone.junctions)
            num_tls = sum(1 for jid in zone.junctions if self.get_junction(jid).type == "traffic_light")
            print(f"  Junctions: {total_junctions} ({num_tls} traffic lights)")

            # Vehicles
            vehicle_type_counts = {}
            stagnant_count = 0
            for vid in zone.current_vehicles:
                vehicle = self.get_vehicle(vid)
                vehicle_type_counts[vehicle.vehicle_type] = vehicle_type_counts.get(vehicle.vehicle_type, 0) + 1
                if vehicle.is_stagnant:
                    stagnant_count += 1

            total_vehicles = len(zone.current_vehicles)
            print(f"  Vehicles: {total_vehicles} ({stagnant_count} stagnant)")
            for vtype, count in vehicle_type_counts.items():
                print(f"    {count} {vtype}")

        print("\n total vehicles in simulation:", len(self.vehicles))
        print("\n-----------------------------------\n")


class SimManager:
    """
    Manages the simulation process using IDs only,
    delegating storage and state to the DataBase.
    """
    def __init__(self, net):
        self.net = net
        self.db = DataBase()
        self.schedule = {}  # step -> list of (vehicle_id, destination)
        self.vehicles_in_route = []  # List of vehicles currently in route
        self.log = Logger().get_logger()
        self.ground_truth_labels = []  # vehicle_id -> trip info
        self.log.info("Simulation Manager initialized.")

    def load_zones(self):
        zone_objects = {}

        # 1. Collect junctions by zone attribute (first!)
        for junction in self.net.getNodes():
            zone_attr = junction.getParam("zone")
            if not zone_attr:
                continue
            zone_id = zone_attr.upper()
            if zone_id not in zone_objects:
                zone_objects[zone_id] = Zone(zone_id)

            junc = Junction(
                junction_id=junction.getID(),
                x=junction.getCoord()[0],
                y=junction.getCoord()[1],
                junc_type=junction.getType(),
                zone=zone_id
            )
            self.db.add_junction(junc)
            zone_objects[zone_id].add_junction(junc.id)

        # 2. Collect edges by zone attribute and update junction connections
        for edge in self.net.getEdges():
            zone_attr = edge.getParam("zone")
            if not zone_attr:
                continue
            zone_id = zone_attr.upper()
            if zone_id not in zone_objects:
                zone_objects[zone_id] = Zone(zone_id)
                print(f"Zone {zone_id} created.")
                self.log.info(f"Zone {zone_id} created.")

            road = Road(
                road_id=edge.getID(),
                from_junction=edge.getFromNode().getID(),
                to_junction=edge.getToNode().getID(),
                speed=edge.getSpeed(),
                length=edge.getLength(),
                num_lanes=len(edge.getLanes()),
                zone=zone_id
            )
            self.db.add_road(road)
            zone_objects[zone_id].add_edge(road.id)
            # print(f"Road {road.id} added to zone {zone_id}.")
            # print(f"Road info: {road.to_dict()}.")

            # Update junction connections ---
            from_junction = self.db.get_junction(road.from_junction)
            to_junction = self.db.get_junction(road.to_junction)
            if from_junction:
                from_junction.add_outgoing(road.id)
            if to_junction:
                to_junction.add_incoming(road.id)

        for zone in zone_objects.values():
            self.db.add_zone(zone)

    def populate_vehicles_from_config(self, config):
        """
        Populates vehicles in the simulation based on a configuration file.
        The configuration file should specify the total number of vehicles,
        the distribution of vehicles across zones, and the types of vehicles.   
        """
        dev_fraction = config.get("vehicle_generation", {}).get("dev_fraction", 1.0)
        orig_total_vehicles = config["vehicle_generation"]["total_num_vehicles"]

        total_vehicles = round(orig_total_vehicles * dev_fraction)
        print(f"Total vehicles to generate: {total_vehicles}")
        self.log.info(f"Total vehicles to generate: {total_vehicles}")
        orig_estimated_peak = self.estimate_required_vehicles(config)
        estimated_peak = round(orig_estimated_peak * dev_fraction)
        print(f"Estimated requiered vehicles: {estimated_peak}")
        self.log.info(f"Estimated requiered vehicles: {estimated_peak}")
        if estimated_peak > total_vehicles:
            print(f"ERROR : Terminating the simulation since estimated requiered vehicles ({estimated_peak}) exceeds total vehicles ({total_vehicles}).")
            self.log.error(f"Terminating the simulation since estimated requiered vehicles ({estimated_peak}) exceeds total vehicles ({total_vehicles}).")  
            exit(1)
        
        zone_alloc = config["vehicle_generation"]["zone_allocation"]
        vehicle_types = config["vehicle_generation"]["vehicle_types"]

        vehicle_id_counter = 0

        # Track per-zone assignments
        active_zone_ids = [zid for zid in zone_alloc if zid.lower() not in ("h", "stagnant")]
        num_zones = len(active_zone_ids)

        sum_vehicles = 0
        # Calculate total stagnant vehicles
        stagnant_per_zone = {}
        total_stagnant = 0
        if "stagnant" in zone_alloc:
            stagnant_pct = zone_alloc["stagnant"]["percentage"]
            total_stagnant = round((stagnant_pct / 100) * total_vehicles)
            base_stag = total_stagnant // num_zones
            extra_stag = total_stagnant % num_zones
            for i, zid in enumerate(active_zone_ids):
                stagnant_per_zone[zid] = base_stag + (1 if i < extra_stag else 0)
                sum_vehicles += stagnant_per_zone[zid]
        
        # Calculate total vehicles in other zones
        active_zone_vehicle_counts = {}
        for zid in zone_alloc:
            if zid.lower() == "stagnant":
                continue
            if zid.upper() == "H":
                continue
            percentage = zone_alloc[zid]["percentage"]
            num_zone_vehicles = round((percentage / 100) * total_vehicles)
            active_zone_vehicle_counts[zid] = num_zone_vehicles
            sum_vehicles += num_zone_vehicles
    
        # Allocate more vehicles to zones until total_vehicles is reached
        # This is to ensure that the total number of vehicles is equal to the specified total
        # vehicles in the configuration
        while total_vehicles > sum_vehicles:
            # Randomly select a zone
            zone_id = random.choice(active_zone_ids)
            active_zone_vehicle_counts[zone_id] += 1
            sum_vehicles += 1
            # print(f"Adding vehicle to zone {zone_id}.")

        
        # Create vehicles in each zone
        # Iterate over each zone and create vehicles
        # based on the specified distribution
        # and vehicle types
        for zone_id, zone_cfg in tqdm(zone_alloc.items(), desc="Allocating vehicles by zone"):
            if zone_id.lower() == "stagnant":
                continue
            if zone_id.upper() == "H":
                continue  # Skip highway zone
            num_zone_vehicles = active_zone_vehicle_counts[zone_id] + stagnant_per_zone.get(zone_id, 0)
            type_distribution = zone_cfg["vehicle_type_distribution"]

            # Get eligible single lanes roads in this zone
            zone = self.db.get_zone(zone_id)
            eligible_roads = [eid for eid in zone.edges if self.db.get_road(eid).num_lanes == 1]
            if not eligible_roads:
                continue

            # Allocate vehicles across types
            type_allocations = {
                vtype: round((vperc / 100) * num_zone_vehicles)
                for vtype, vperc in type_distribution.items()
            }
            
            vehicle_specs = [
                (vtype, vehicle_types[vtype].copy())
                for vtype, count in type_allocations.items()
                for _ in range(count)
                ]
            random.shuffle(vehicle_specs)

            per_road = len(vehicle_specs) // len(eligible_roads)
            overflow = len(vehicle_specs) % len(eligible_roads)

            vehicle_iter = iter(vehicle_specs)

            for i, road_id in enumerate(tqdm(eligible_roads, desc=f"Zone {zone_id} roads")):
                vehicles_on_road = per_road + (1 if i < overflow else 0)
                road = self.db.get_road(road_id)
                net_edge = self.net.getEdge(road_id)
                length = road.length
                spacing = length / (vehicles_on_road + 1)

                for j in range(vehicles_on_road):
                    if len(self.db.vehicles) >= total_vehicles:
                        break
                    try:
                        vtype, vcfg = next(vehicle_iter)
                    except StopIteration:
                        break

                    pos = (j + 1) * spacing
                    x, y = net_edge.getLane(0).getShape()[0]

                    vehicle = Vehicle(
                        vehicle_id=f"veh_{vehicle_id_counter}",
                        vehicle_type=vtype,
                        current_zone=zone_id,
                        current_edge=road_id,
                        current_position=pos,
                        current_x=x,
                        current_y=y,
                        length=vcfg["length"],
                        width=vcfg["width"],
                        height=vcfg["height"],
                        color=vcfg["color"],
                        status="parked",
                        is_stagnant=False
                    )

                    self.db.add_vehicle(vehicle)
                    road.add_vehicle_and_update(vehicle)
                    zone.add_original_vehicle(vehicle.id)
                    zone.add_current_vehicle(vehicle.id)

                    self.assign_destinations(vehicle, self.db.zones, config["landmarks"])

                    vehicle_id_counter += 1

                    # print(f"Added {vtype} vehicle {vehicle.id} in zone {zone_id} on road {road_id} at position {pos:.2f}")

            # Randomly convert some vehicles in this zone to stagnant
            zone_vehicle_ids = list(zone.current_vehicles)
            if 'stagnant' in zone_alloc:
                num_to_convert = stagnant_per_zone.get(zone_id, 0)
                to_convert = random.sample(zone_vehicle_ids, min(num_to_convert, len(zone_vehicle_ids)))
                for vid in to_convert:
                    v = self.db.get_vehicle(vid)
                    v.is_stagnant = True
                    v.color = "white"
                    # print(f"Converted vehicle {vid} to stagnant in zone {zone_id}")
    
        print(f"Zones loaded: {len(self.db.zones)}")
        self.log.info(f"Zones loaded: {len(self.db.zones)}")

        for zid, zone in self.db.zones.items():
            print(f"Zone {zid}: {len(zone.edges)} roads, {len(zone.junctions)} junctions")
            self.log.info(f"Zone {zid}: {len(zone.edges)} roads, {len(zone.junctions)} junctions")

        self.db.print_zone_statistics()

    def clear_roads_and_zones(self):
        """
        Clears all roads and zones vehicles before dispatching.
        """
        for road in self.db.roads.values():
            road.vehicles_on_road.clear()
            road.density = 0.0
            road.avg_speed = 0.0

        for zone in self.db.zones.values():
            zone.original_vehicles.clear()
            zone.current_vehicles.clear()

    def assign_destinations(self, vehicle, zone_map, landmark_map):

        # HOME
        vehicle.destinations["home"] = {
            "edge": vehicle.current_edge,
            "position": vehicle.current_position
        }

        # WORK (random edge in Zone B)
        zone_b_edges = [eid for eid in zone_map["B"].edges if self.db.get_road(eid).num_lanes == 1]
        work_edge = random.choice(zone_b_edges)
        vehicle.destinations["work"] = {
            "edge": work_edge,
            "position": random.uniform(1.0, self.db.get_road(work_edge).length - 1.0)
        }

        # FRIEND 1: same zone, single-lane
        same_zone_edges = [eid for eid in zone_map[vehicle.current_zone].edges if self.db.get_road(eid).num_lanes == 1]
        friend1_edge = random.choice(same_zone_edges)
        vehicle.destinations["friend1"] = {
            "edge": friend1_edge,
            "position": random.uniform(1.0, self.db.get_road(friend1_edge).length - 1.0)
        }

        # FRIEND 2 & 3: in other zones, single-lane only
        other_zones = [z for z in zone_map if z != vehicle.current_zone and z != "H"]
        for i in range(2, 4):
            other_zone_id = other_zones[i - 2]
            eligible_edges = [eid for eid in zone_map[other_zone_id].edges if self.db.get_road(eid).num_lanes == 1]
            if not eligible_edges:
                continue
            other_edge = random.choice(eligible_edges)
            vehicle.destinations[f"friend{i}"] = {
                "edge": other_edge,
                "position": random.uniform(1.0, self.db.get_road(other_edge).length - 1.0)
            }

        # PARKS 1–4 (Zone A)
        for i in range(1, 5):
            park_edge = random.choice(landmark_map[f"park{i}"])
            vehicle.destinations[f"park{i}"] = {
                "edge": park_edge,
                "position": random.uniform(1.0, self.db.get_road(park_edge).length - 1.0)
            }

        # STADIUMS 1–2 (Zone C)
        for i in range(1, 3):
            stadium_edge = random.choice(landmark_map[f"stadium{i}"])
            vehicle.destinations[f"stadium{i}"] = {
                "edge": stadium_edge,
                "position": random.uniform(1.0, self.db.get_road(stadium_edge).length - 1.0)
            }

        # RESTAURANTS by zone
        rest_map = {"A": "restaurantA", "B": "restaurantB", "C": "restaurantC"}
        for zone_id, label in rest_map.items():
            eligible = [eid for eid in zone_map[zone_id].edges if self.db.get_road(eid).num_lanes == 1]
            if eligible:
                edge = random.choice(eligible)
                vehicle.destinations[label] = {
                    "edge": edge,
                    "position": random.uniform(1.0, self.db.get_road(edge).length - 1.0)
                }

    def add_to_schedule(self, step, trips):
        """
        Adds a list of (vehicle_id, destination_label) to be dispatched at a specific simulation step.
        """
        if step not in self.schedule:
            self.schedule[step] = []
        self.schedule[step].extend(trips)

    def dispatch(self, current_step, traci):
        """
        Dispatches all vehicles scheduled for the given simulation step.
        """
        if current_step not in self.schedule:
            return
        
        curr_week = current_step // 604800

        for vehicle_id, origin_label, destination_label in self.schedule[current_step]:
            vehicle = self.db.get_vehicle(vehicle_id)
            if vehicle.status != "parked":
                continue  # Only dispatch parked vehicles
            origin = vehicle.destinations[origin_label]
            
            # set current vehicle properties
            vehicle.current_edge = origin["edge"]
            vehicle.current_position = float(origin["position"])
            vehicle.current_zone = self.db.get_road(vehicle.current_edge).zone
            # set origin properties
            vehicle.origin_name = origin_label
            vehicle.origin_edge = origin["edge"]
            vehicle.origin_position = float(origin["position"])
            vehicle.origin_zone = vehicle.current_zone
            vehicle.origin_step = current_step+1
         
            destination = vehicle.destinations[destination_label]   
            vehicle.destination_name = destination_label
            vehicle.destination_edge = destination["edge"]
            vehicle.destination_position = float(destination["position"])
            vehicle.destination_zone = self.db.get_road(vehicle.destination_edge).zone

            route_id = f"route_{vehicle_id}_to_{destination_label}_{curr_week}"

            try:
                route_result = traci.simulation.findRoute(vehicle.current_edge, destination["edge"])
                full_route_edges = route_result.edges
                traci.route.add(routeID=route_id, edges=full_route_edges)
                vehicle.route = list(full_route_edges)
                vehicle.route_left = list(full_route_edges)
                vehicle.route_length = round(sum(
                    self.db.get_road(e).length for e in vehicle.route_left if self.db.get_road(e)), 2)
                vehicle.route_length_left = vehicle.route_length

                traci.vehicle.add(
                                vehID=vehicle.id,
                                routeID=route_id,
                                typeID=vehicle.vehicle_type,
                                depart=vehicle.origin_step,
                                departPos=vehicle.current_position,
                                departSpeed=0,
                                departLane="0",
                            )
                if vehicle.is_stagnant:
                    traci.vehicle.setColor(vehicle.id, (255, 255, 255))  # White for stagnant vehicles
                vehicle.origin_x, vehicle.origin_y = traci.simulation.convert2D(vehicle.origin_edge, float(vehicle.origin_position), 0)
                vehicle.current_x, vehicle.current_y = vehicle.origin_x, vehicle.origin_y
                vehicle.destination_x, vehicle.destination_y = traci.simulation.convert2D(vehicle.destination_edge, float(vehicle.destination_position), 0)
                road = self.db.get_road(vehicle.current_edge)
                if road:
                    road.add_vehicle_and_update(vehicle)
                else:
                    self.log.fatal(f"Road {vehicle.current_edge} not found in database. Cannot add vehicle {vehicle.id}.")
                    exit(1)
                traci.vehicle.subscribe(vehicle.id, [tc.VAR_ROAD_ID])
                self.vehicles_in_route.append(vehicle.id)
                self.log.info(f"[DISPATCHED] {vehicle.id} from {origin_label} to {destination_label} at step {vehicle.origin_step}={self.convert_seconds_to_time(vehicle.origin_step)} distance {vehicle.route_length}m.")
                vehicle.status = "in_route"
            except traci.TraCIException as e:
                print(f"[ERROR] Failed to dispatch {vehicle.id}: {e}")

        del self.schedule[current_step]

    def schedule_from_config(self, config):
        schedule_entries = config.get("weekday_schedule", [])
        num_weeks = config["vehicle_generation"]["simulation_weeks"]
        dev_fraction = config.get("vehicle_generation", {}).get("dev_fraction", 1.0)
        seconds_in_day = 86400
        seconds_in_week = seconds_in_day * 7
        num_scheduled_vehicles = 0
        
        # Create a schedule for each week
        for week in tqdm(range(num_weeks), desc="Scheduling vehicles by week"):
            week_start = week * seconds_in_week
            num_weekly_vehicles = 0
            # Adjust the start and end times for the current week       
            for entry in tqdm(schedule_entries, desc=f"Week {week+1} entries", leave=False):
                self.log.info(f"Scheduling vehicles for entry: {entry.get('name', 'Unnamed')}")
                start_sec = self.convert_time_to_seconds(entry["start_time"]) + week_start
                end_sec = self.convert_time_to_seconds(entry["end_time"]) + week_start
                vpm_rate = entry.get("vpm_rate", 0)
                vpm_rate = vpm_rate * dev_fraction
                self.log.info(f"VPM rate: {vpm_rate}")
                source_zones = entry.get("source_zones", [])
                origin_keys = entry.get("origin", [])
                destination_keys = entry.get("destination", [])
                repeat_days = entry.get("repeat_on_days", [])

                # Compute interval between dispatches
                interval = max(60 // vpm_rate, 1)
                local_steps = list(range(start_sec, end_sec + 1, int(interval)))
                if len(local_steps) == 0:
                    local_steps = [start_sec]
                for day in repeat_days:
                    num_scheduled_vehicles_per_day = 0
                    base_step = (day - 1) * seconds_in_day
                    steps = [base_step + s for s in local_steps]

                    for zone_id in source_zones:
                        zone = self.db.get_zone(zone_id)
                        eligible = [
                            v for v in zone.current_vehicles
                            if not self.db.get_vehicle(v).scheduled[week]
                        ]
                        eligible = eligible[:int(len(eligible) * dev_fraction)]
                        steps_to_use = steps[:int(len(eligible))]
                        for step, veh_id in zip(steps_to_use, eligible):
                            vehicle = self.db.get_vehicle(veh_id)
                            vehicle.scheduled[week] = True

                            while True:
                                origin = random.choice(origin_keys)
                                dest = random.choice(destination_keys)
                                if origin != dest:
                                    break
                                else:
                                    self.log.warning(f"Origin {origin} and destination {dest} are the same. Retrying...")
            
                            self.add_to_schedule(step, [(veh_id, origin, dest)])
                            num_scheduled_vehicles += 1
                            num_weekly_vehicles += 1
                            num_scheduled_vehicles_per_day += 1
                    if num_scheduled_vehicles_per_day == 0:
                        print(f"No vehicles scheduled for dispatch on day {day} in week {week + 1}.")
                        self.log.error(f"No vehicles scheduled for dispatch on day {day} in week {week + 1}.")
                        exit(1)

                    self.log.info(f"Week {week + 1} scheduled vehicles for day {day}: {num_scheduled_vehicles_per_day}")
            self.log.info(f"Week {week + 1} scheduled vehicles: {num_weekly_vehicles}")
        self.log.info(f"Total vehicles scheduled for dispatch: {num_scheduled_vehicles}")
        self.log.info("All schedules created.") 
        self.log.info(f"Total scheduled vehicles: {sum(len(v) for v in self.schedule.values())}")
        self.log.info(f"Total vehicles in simulation: {len(self.db.vehicles)}")

    def get_vehicles_in_route(self):
        """
        Returns a list of vehicles that are currently in route.
        """
        return self.vehicles_in_route
    
    def convert_time_to_seconds(self, time_str):
        """
        Converts a time string formatted as HH:MM into the number of seconds since midnight.
        """
        hour, minute = map(int, time_str.split(":"))
        return hour * 3600 + minute * 60
    
    def convert_seconds_to_time(self, seconds):
        """
        Converts a number of seconds  into a time string formatted as WW:DD:HH:MM:SS.
        """
        week = seconds // 604800
        day = (seconds % 604800) // 86400
        hour = (seconds % 86400) // 3600
        minute = (seconds % 3600) // 60
        second = seconds % 60
        return f"{week:02}:{day:02}:{hour:02}:{minute:02}:{second:02}"
    
    def estimate_peak_vehicles(self, config):
        seconds_per_day = 86400
        timeline = defaultdict(int)

        for task in config.get("weekday_schedule", []):
            start = self.convert_time_to_seconds(task["start_time"])
            end = self.convert_time_to_seconds(task["end_time"])
            duration = end - start
            vpm = float(task.get("vpm_rate", 0.1))
            interval = 60 / vpm

            repeat_days = task.get("repeat_on_days", [1, 2, 3, 4, 5])
            return_delay = task.get("return_after_seconds", 0)

            for day in repeat_days:
                base_step = (day - 1) * seconds_per_day
                departure_steps = range(start, end, int(interval))

                for dep in departure_steps:
                    timeline[base_step + dep] += 1
                    timeline[base_step + dep + return_delay] -= 1

        # Compute the peak concurrent vehicle count
        running_total = 0
        max_vehicles = 0
        for step in sorted(timeline.keys()):
            running_total += timeline[step]
            max_vehicles = max(max_vehicles, running_total)

        return max_vehicles
    
    def estimate_required_vehicles(self, config):
        total = 0
        seconds_per_day = 86400

        for entry in config.get("weekday_schedule", []):
            vpm = entry.get("vpm_rate", 0.0)
            interval = 60 / max(vpm, 0.01)

            start = self.convert_time_to_seconds(entry["start_time"])
            end = self.convert_time_to_seconds(entry["end_time"])
            duration = max(end - start, 0)

            num_dispatches = int(duration // interval)
            num_zones = len(entry.get("source_zones", []))
            num_days = len(entry.get("repeat_on_days", [1, 2, 3, 4, 5]))

            total += num_dispatches * num_zones * num_days

        return total
    
    def calculate_simulation_limit(self, config):
        # Calculate the total simulation time limit based on the configuration
        # Allow for an additional 30 minutes for finalization
        extra_time = 1800  # 30 minutes in seconds
        num_weeks = config["vehicle_generation"]["simulation_weeks"]
        seconds_in_day = 86400
        seconds_in_week = seconds_in_day * 7
        return num_weeks * seconds_in_week + extra_time 

    def update(self, current_step, traci):
        for vid in traci.vehicle.getIDList():
            vehicle = self.db.vehicles[vid]
            if vehicle is not None and vehicle.status == "in_route":
                current_edge = traci.vehicle.getRoadID(vid)
                curr_road = self.db.get_road(current_edge)
                if curr_road is None:
                    continue
                # update vehicle dynamic features
                vehicle.current_position = traci.vehicle.getLanePosition(vid)
                vehicle.acceleration = traci.vehicle.getAcceleration(vid)
                vehicle.speed = traci.vehicle.getSpeed(vid)
                vehicle.current_lane = traci.vehicle.getLaneID(vid)
                vehicle.current_x, vehicle.current_y = traci.vehicle.getPosition(vid)
                vehicle.current_zone = curr_road.zone
                
                prev_road = self.db.get_road(vehicle.current_edge)
                prev_edge = vehicle.current_edge

                # only update vehicle in road for average road speed and density 
                if(current_edge == prev_edge):
                    curr_road.add_vehicle_and_update(vehicle)
                else:
                    # if vehicle changed road, update its state
                    self.log.info(f"Vehicle {vehicle.id} changed road from {vehicle.current_edge} to {current_edge}.")
                    # print(f"Vehicle {vehicle.id} changed road from {vehicle.current_edge} to {current_edge}.")
                
                # remove vehicle from previous Road and add to current Road
                if prev_road is not None and prev_road != curr_road:
                    # print(f"Vehicle {vehicle.id} is moving from {vehicle.current_edge} to {current_edge}.")
                    self.log.info(f"Vehicle {vehicle.id} is moving from {vehicle.current_edge} to {current_edge}.")
                    if vehicle.id in prev_road.vehicles_on_road.keys():
                        prev_road.remove_vehicle_and_update(vehicle)
                        curr_road.add_vehicle_and_update(vehicle)
                    else:
                        print(f"Previous road {vehicle.current_edge} not found in prev_road.vehicles_on_road.")
                        self.log.error(f"Previous road {vehicle.current_edge} not found in prev_road.vehicles_on_road.")
                
                # update vehicle route
                vehicle.current_edge = current_edge
                if vehicle.route and vehicle.current_edge in vehicle.route:
                    idx = vehicle.route.index(vehicle.current_edge)
                    vehicle.route_left = vehicle.route[idx:]
                    vehicle.route_length_left = round(sum(self.db.get_road(e).length for e in vehicle.route_left if self.db.get_road(e)), 2)
                
               
                if current_edge == vehicle.destination_edge:
                    vehicle.status = "parked"
                    vehicle.current_edge = vehicle.destination_edge
                    vehicle.current_position = vehicle.destination_position
                    vehicle.route_length_left = 0
                    vehicle.destination_step = current_step
                    self.vehicles_in_route.remove(vid)
                    curr_road.remove_vehicle_and_update(vehicle)
                    self.add_ground_truth_label(vehicle)
                    self.log.info(f"Vehicle {vehicle.id} arrived at destination {vehicle.destination_name} at {self.convert_seconds_to_time(current_step)} started {self.convert_seconds_to_time(vehicle.origin_step)} duration {self.convert_seconds_to_time(current_step - vehicle.origin_step)}")
                    if current_step - vehicle.origin_step > 2 * 60 * 60 :
                        self.log.warning(f"Vehicle {vehicle.id} travel duration {self.convert_seconds_to_time(current_step - vehicle.origin_step)} exceeds top limit")
                        print(f"Vehicle {vehicle.id} travel duration {self.convert_seconds_to_time(current_step - vehicle.origin_step)} exceeds top limit")
        if current_step % 300 == 0:
            self.log.info(f"step {current_step} time {self.convert_seconds_to_time(current_step)} vehicles in route: {len(self.vehicles_in_route)}") 

    def save_snapshot(self, snapshot_dir, step):
        
        # Gather all node features (vehicles + junctions)
        nodes = [v.to_dict() for v in self.db.vehicles.values() if v.status == "in_route"]
        nodes += [j.to_dict() for j in self.db.junctions.values()]
        
        # Gather all edge features (roads)
        edges = [r.to_dict() for r in self.db.roads.values()]

        snapshot = {
            "step": step,
            "nodes": nodes,
            "edges": edges
        }
        # Filename: e.g., step_000123.json
        filename = os.path.join(snapshot_dir, f"step_{step:06d}.json")
        with open(filename, "w") as f:
            json.dump(snapshot, f, indent=2)

    def add_ground_truth_label(self, vehicle):
        trip = {
            "vehicle_id": vehicle.id,
            "origin_time_sec": vehicle.origin_step,
            "destination_time_sec": vehicle.destination_step,
            "origin_x": vehicle.origin_x,
            "origin_y": vehicle.origin_y,
            "destination_x": vehicle.destination_x,
            "destination_y": vehicle.destination_y,
            "origin_edge": vehicle.origin_edge,
            "destination_edge": vehicle.destination_edge,
            "route": vehicle.route,
            "initial_route_length": vehicle.route_length,
            "total_travel_time_seconds": (vehicle.destination_step - vehicle.origin_step)
        }
        self.ground_truth_labels.append(trip)
    
    def save_labels_file(self, filename="ground_truth.json"):
        with open(filename, "w") as f:
            json.dump(self.ground_truth_labels, f, indent=2)

    def generate_id_mapping_files(self, mapping_dir="mappings"):
        """
        Generates mapping files for vehicles, junctions, and roads.
        Each file contains a mapping from IDs to their respective attributes.
        """
        os.makedirs(mapping_dir, exist_ok=True)

        # Vehicles
        vehicle_mapping = {v.id: v.to_dict() for v in self.db.vehicles.values()}
        with open(os.path.join(mapping_dir, "vehicle_mapping.json"), "w") as f:
            json.dump(vehicle_mapping, f, indent=2)

        # Junctions
        junction_mapping = {j.id: j.to_dict() for j in self.db.junctions.values()}
        with open(os.path.join(mapping_dir, "junction_mapping.json"), "w") as f:
            json.dump(junction_mapping, f, indent=2)

        # Roads
        edge_mapping = {r.id: r.to_dict() for r in self.db.roads.values()}
        with open(os.path.join(mapping_dir, "edge_mapping.json"), "w") as f:
            json.dump(edge_mapping, f, indent=2)
        
        print(f"ID mappings saved to {mapping_dir}/")
        print(f"Total vehicles: {len(self.db.vehicles)}")
        print(f"Total junctions: {len(self.db.junctions)}")
        print(f"Total roads: {len(self.db.roads)}")