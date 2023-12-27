import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, box, MultiPoint, MultiLineString
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from shapely.ops import nearest_points
from shapely.ops import nearest_points
import heapq
import itertools
from rtree import index
"""Script to identify and find shortest paths using network distance and augmented Knearest algorithm from a given hydrant to nearest network neighboring hydrants.
   Nearest neighboring hydrants should be identified using djikstra network distance.
   A Shortest path should never traverse through one hydrant to reach another, and nearest network neighboring hydrants should always be prioritized over farther ones.
   Each shortest path should return a source node, a target node, and a length of the path. Each path's source and target should be a hydrant id, as each path starts and ends with a hydrant."""


def assign_ids_to_snapped_coords(snapped_coords, hydrant_shapefile):
    coord_id_map = {}
    for coord in snapped_coords:
        snapped_point = Point(coord)
        # Find the hydrant in the shapefile that is closest to the snapped_point
        closest_hydrant = hydrant_shapefile.distance(snapped_point).idxmin()
        hydrant_row = hydrant_shapefile.iloc[closest_hydrant]
        # Assign the id from the closest hydrant to the snapped coordinate
        hydrant_id = (
            hydrant_row["id"]
        )
        coord_id_map[coord] = hydrant_id
    return coord_id_map

def create_network_snapped(roads, hydrants, search_radius):
    G = nx.MultiGraph()
    snapped_coords = []
    # Create a GeoDataFrame with all the vertices in the road network
    vertices = []
    for _, road in roads.iterrows():
        vertices.extend(Point(round(coord[0], 3), round(coord[1], 3)) for coord in road.geometry.coords)
    vertices_gdf = gpd.GeoDataFrame(geometry=vertices)
    # Iterate over all hydrants
    for _, hydrant in tqdm(hydrants.iterrows()):
        hydrant_point = hydrant.geometry
        # Create a bounding box around the hydrant point
        bounding_box = box(
            hydrant_point.x - search_radius,
            hydrant_point.y - search_radius,
            hydrant_point.x + search_radius,
            hydrant_point.y + search_radius,
        )
        # Find all vertices within the bounding box (preliminary filter)
        candidates = vertices_gdf[vertices_gdf.intersects(bounding_box)]
        # Find the nearest vertex within the candidates
        nearest_vertex = None
        min_distance = float("inf")
        for _, candidate in candidates.iterrows():
            distance = hydrant_point.distance(candidate.geometry)
            if distance < min_distance:
                min_distance = distance
                nearest_vertex = candidate.geometry.coords[0]
        if nearest_vertex:
            snapped_coords.append(nearest_vertex)
    # Add road segments to the graph
    for _, road in roads.iterrows():
        for i in range(len(road.geometry.coords) - 1):
            start = (round(road.geometry.coords[i][0], 3), round(road.geometry.coords[i][1], 3))
            end = (round(road.geometry.coords[i+1][0], 3), round(road.geometry.coords[i+1][1], 3))
            G.add_edge(start, end, weight=LineString([start, end]).length)
    # Add snapped hydrants as nodes
    for coord in snapped_coords:
        G.add_node(coord)
    return G, snapped_coords


def connect_components(G):
    # Identify all connected components
    components = list(nx.connected_components(G))
    # Find the largest connected component
    largest_component = max(components, key=len)
    # For each of the other components
    for component in components:
        if component != largest_component:
            # Find the node in the component that is closest to any node in the largest component
            closest_nodes = nearest_points(
                MultiPoint([Point(node) for node in component]),
                MultiPoint([Point(node) for node in largest_component])
            )
            # Get the nodes corresponding to the closest points
            source_node = [node for node in component if Point(node) == closest_nodes[0]][0]
            target_node = [node for node in largest_component if Point(node) == closest_nodes[1]][0]
            # Add an edge between these two nodes
            G.add_edge(source_node, target_node)
    return G


def find_k_nearest_neighbors(G, hydrants, roads, snapped_coords, k, coord_id_map):
    # Ensure the graph is undirected for bidirectional traversal
    G = G.to_undirected()
    # Initialize dictionaries for storing shortest paths and neighborhoods
    shortest_paths = {}
    neighborhoods = {}
    # Set of all hydrant locations for easy lookup
    hydrant_nodes = set(snapped_coords)

    # Iterate through each hydrant coordinate
    for i in tqdm(range(len(snapped_coords))):
        F_node = snapped_coords[i]  # Current source hydrant
        F_node_id = coord_id_map.get(F_node, F_node)  # Get ID of the source hydrant

        # Use Dijkstra's algorithm to find shortest paths from this hydrant to all others
        lengths, paths = nx.single_source_dijkstra(G, F_node, weight="weight")

        valid_paths = []  # Store paths that meet the criteria

        # Filter paths to include only those targeting other hydrants and exclude paths through intermediate hydrants
        for target_node, path_length in lengths.items():
            if target_node in hydrant_nodes and target_node != F_node:
                path = paths[target_node]
                if not any(node in hydrant_nodes - {F_node, target_node} for node in path):
                    rounded_length = round(path_length, 2)  # Round path length to 2 decimal places
                    valid_paths.append((rounded_length, target_node, path))

        valid_paths.sort(key=lambda x: x[0])  # Sort valid paths by length
        selected_paths = valid_paths[:1]  # Start with the absolute nearest neighbor

        # Track edges used by the nearest neighbor to avoid reusing them
        used_edges = set()
        if selected_paths:
            for j in range(len(selected_paths[0][2]) - 1):
                used_edges.add((selected_paths[0][2][j], selected_paths[0][2][j + 1]))

        # Ensure at least one path uses a different edge than those used by the nearest neighbor
        for path_length, target_node, path in valid_paths[1:]:
            if len(selected_paths) >= k:
                break

            new_edges = set((path[j], path[j + 1]) for j in range(len(path) - 1))
            if not used_edges & new_edges:
                selected_paths.append((path_length, target_node, path))
                break

        # Continue finding up to 'k' nearest neighbors, avoiding redundant paths
        for path_length, target_node, path in valid_paths[len(selected_paths):]:
            if len(selected_paths) >= k:
                break
            if not any(node in hydrant_nodes - {F_node, target_node} for node in path):
                selected_paths.append((path_length, target_node, path))

        # Map each selected path to the neighborhood of the source hydrant
        neighborhoods[F_node_id] = [coord_id_map.get(path[2][-1], path[2][-1]) for path in selected_paths if coord_id_map.get(path[2][-1], path[2][-1]) != F_node_id]
        # Store detailed information about each path
        for path_length, target_node, path in selected_paths:
            target_id = coord_id_map.get(target_node, target_node)
            path_segments = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            hydrant_ids_on_path = [coord_id_map.get(node, node) for node in path if node in hydrant_nodes]

            path_info = {
                "source": F_node_id,
                "target": target_id,
                "length": path_length,
                "path": path_segments,
                "hydrant_ids": hydrant_ids_on_path
            }

            if F_node_id not in shortest_paths:
                shortest_paths[F_node_id] = []
            shortest_paths[F_node_id].append(path_info)

    return shortest_paths, neighborhoods

def count_neighborhoods(neighborhoods, shortest_paths):
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for neighbors in neighborhoods.values():
        if len(neighbors) in counts:
            counts[len(neighbors)] += 1

    for source_id, paths in shortest_paths.items():
        for path_info in paths:
            target_id = path_info["target"]
            if target_id == source_id:
                print(f"Source and target are the same: {source_id}")

    return counts



def visualize_network(G, roads, original_hydrants, snapped_coords, nearest_neighbors=None):
    # Convert snapped coords to GeoDataFrame
    snapped_hydrants = gpd.GeoDataFrame(
        geometry=[Point(coord) for coord in snapped_coords]
    )
    fig, ax = plt.subplots(figsize=(12, 12))
    roads.plot(ax=ax, color="gray", linewidth=1)
    snapped_hydrants.plot(ax=ax, color="red", markersize=50, label="Snapped Hydrants")
    # Plot the graph
    for edge in G.edges():
        start, end = edge
        ax.plot([start[0], end[0]], [start[1], end[1]], color="blue", alpha=0.5)
    # Plot shortest paths
    if nearest_neighbors:
        for _, paths in nearest_neighbors.items():
            for path_info in paths:
                for segment in path_info["path"]:
                    start, end = segment
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color="green",
                        linewidth=2,
                    )
    plt.legend()
    plt.show()



def paths_to_linestrings(nearest_neighbors):
    linestrings = []
    lengths = []
    sources = []
    targets = []
    # Iterate over nearest neighbors to create linestrings and capture attributes
    for start_id, paths in nearest_neighbors.items():
        for path_info in paths:
            end_id = path_info["target"]
            path_nodes = path_info["path"]
            flat_path_nodes = [point for segment in path_nodes for point in segment]
            linestrings.append(LineString(flat_path_nodes))
            lengths.append(path_info["length"])
            sources.append(start_id)
            targets.append(end_id)
    # Create a GeoDataFrame with the linestrings and their attributes
    linestrings_gdf = gpd.GeoDataFrame(
        {"source": sources, "target": targets, "length": lengths, "geometry": linestrings}
    )

    return linestrings_gdf


def visualize_connected_components(G, roads, snapped_coords):
    # Identify connected components
    connected_components = list(nx.connected_components(G))
    # Create a GeoDataFrame for snapped hydrants
    snapped_hydrants_gdf = gpd.GeoDataFrame(geometry=[Point(coord) for coord in snapped_coords])
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot roads for context
    roads.plot(ax=ax, color="gray", linewidth=1, label="Roads")
    # Define a color palette
    color_cycle = itertools.cycle(plt.cm.tab20.colors)  # Color cycle for different components

    # Iterate over connected components and plot them with different colors
    for component in connected_components:
        component_color = next(color_cycle)

        # Plot edges for each connected component
        subgraph = G.subgraph(component)
        for edge in subgraph.edges():
            start, end = edge 
            ax.plot([start[0], end[0]], [start[1], end[1]], color=component_color, linewidth=2)

        # Plot hydrants in the component
        component_hydrants = [coord for coord in snapped_coords if coord in component]
        component_hydrants_gdf = gpd.GeoDataFrame(geometry=[Point(coord) for coord in component_hydrants])
        component_hydrants_gdf.plot(ax=ax, color=component_color, markersize=50, alpha=0.5)

    # Legend and labels
    plt.title("Network with Different Connected Components")
    plt.legend()
    plt.show()


def save_neighborhoods_to_shapefile(neighborhoods, shortest_paths, coord_id_map, snapped_coords, output_path1):
    hydrant_ids = []
    neighbors_list = []
    path_lengths_list = []
    geometries = []

    for hydrant_id, neighbors in neighborhoods.items():
        hydrant_ids.append(hydrant_id)
        neighbors_str = ", ".join(map(str, neighbors))
        neighbors_list.append(neighbors_str)

        # Extract and store path lengths with target IDs
        lengths = ["{}: {}".format(coord_id_map[shortest_paths[hydrant_id][i]['target']], shortest_paths[hydrant_id][i]['length']) 
           for i in range(len(neighbors)) if i in shortest_paths[hydrant_id]]

        # Find the corresponding snapped coordinate for the hydrant_id
        for coord, id in coord_id_map.items():
            if id == hydrant_id:
                geometry = Point(coord)
                geometries.append(geometry)
                break

        # Extract and store path lengths
        lengths = [shortest_paths[hydrant_id][i]['length'] for i in range(len(neighbors))]
        path_lengths_list.append(", ".join(map(str, lengths)))

    gdf = gpd.GeoDataFrame({
        'hydrant_id': hydrant_ids,
        'neighbors': neighbors_list,
        'path_lengths': path_lengths_list,  # Add path lengths here
        'geometry': geometries
    }, geometry='geometry')
    gdf.crs = "EPSG:2229"  
    gdf.to_file(output_path1)

    return gdf



def main():
    # Load road and hydrant data
    road_path = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\Road centerlines\sub_centerlines_splited.shp"
    hydrants_path = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\hydrants\hydrants_reprojec_to_roads.shp"
    roads = gpd.read_file(road_path).reset_index(drop=True)
    hydrants = gpd.read_file(hydrants_path).reset_index(drop=True)
    hydrant_shapefile_path = (
        r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\hydrants\drive-download-20231202T013905Z-001\hydrants_whole_region.shp"
    )
    hydrant_shapefile = gpd.read_file(hydrant_shapefile_path)
    print(hydrant_shapefile.columns)
    G, snapped_coords = create_network_snapped(roads, hydrants, search_radius=5)
    # Assign IDs to snapped coordinates
    coord_id_map = assign_ids_to_snapped_coords(snapped_coords, hydrant_shapefile)
    print("assigned af")
    # Connect all disconnected components to the largest component
    #G = connect_components(G)
    # Find shortest paths using the IDs
    shortest_paths, neighborhoods =find_k_nearest_neighbors(G, hydrants, roads, snapped_coords, 4, coord_id_map)

    print("# of paths", len(shortest_paths))
    print("# of hoods", len(neighborhoods))
    counts = count_neighborhoods(neighborhoods, shortest_paths)
    print(counts)
    first_5_neighborhoods = list(neighborhoods.items())[:5]
    for start_node, neighbors in first_5_neighborhoods:
        print(f"Start node: {start_node}, Neighbors: {neighbors}")
    # Get the GeoDataFrame with shortest path linestrings
    linestrings_gdf = paths_to_linestrings(shortest_paths)
    print("o")
    # Visualize the results
    visualize_network(G, roads, hydrants, snapped_coords, shortest_paths)
   
    # Save shortest paths to a new shapefile
    # Assuming paths_to_linestrings now returns a GeoDataFrame with all the necessary data
    visualize_connected_components(G, roads, snapped_coords)
    # Now you don't need to create a new GeoDataFrame, as linestrings_gdf already is one
    output_path = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\outputs\K_paths.shp"
    output_path1 = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\outputs\K_neighborhoods.shp"
   
    save_neighborhoods_to_shapefile(neighborhoods, shortest_paths, coord_id_map, snapped_coords, output_path1)
    linestrings_gdf.to_file(output_path)
    print("All done")


if __name__ == "__main__":
    main()
