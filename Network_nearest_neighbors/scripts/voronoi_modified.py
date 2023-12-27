import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, box, MultiPoint, Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from shapely.ops import nearest_points, split
from shapely.ops import nearest_points
import heapq
import itertools
from collections import defaultdict
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt


"""Creates Voronoi Polygons around the snapped hydrants and uses them to find the nearest neighbors of each hydrant. Neighbors are defined as hydrants
 that share a Voronoi edge with the hydrant of interest. The shortest path between each hydrant and its neighbors is then found using Dijkstra's algorithm. If a neighbors
 shortest path contains a hydrant, the path is split at that hydrant and the path is recalculated. The final output is a shapefile containing the hydrant ID, the IDs of its neighbors,
 and  the length of the shortest paths. The output also contains a shapefile of the shortest paths."""








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
    for _, road in tqdm(roads.iterrows()):
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
def voronoi_polygons(voronoi, diameter):
    """Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.
   
    stack overflow code by Gareth Res: https://stackoverflow.com/questions/23901943/voronoi-compute-exact-boundaries-of-every-region
    """
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            # Infinite ridge starting at ridge point with index v,
            # equidistant from input points with indexes p and q.
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)




    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))








def find_all_nearest_neighbors_voronoi(G, hydrants, roads, snapped_coords, coord_id_map):
    G = G.to_undirected()
    shortest_paths = {}
    neighborhoods = {}
    hydrant_nodes = set(snapped_coords)


    vor = Voronoi(snapped_coords)
    diameter = np.linalg.norm(np.ptp(snapped_coords, axis=0))
    voronoi_polys = list(voronoi_polygons(vor, diameter))


    for i in tqdm(range(len(snapped_coords))):
        F_node = snapped_coords[i]
        F_node_id = coord_id_map.get(F_node, F_node)

        neighbors = set([snapped_coords[j] for j in range(len(snapped_coords)) if voronoi_polys[i].touches(voronoi_polys[j])])

        valid_paths = []

        for target_node in neighbors:
            if target_node != F_node:
                try:
                    path_length = nx.dijkstra_path_length(G, F_node, target_node, weight="weight")
                    path = nx.dijkstra_path(G, F_node, target_node, weight="weight")
                    intermediate_nodes = [node for node in path if node in hydrant_nodes - {F_node, target_node}]
                    if intermediate_nodes:
                        target_node = intermediate_nodes[0]
                        path = nx.dijkstra_path(G, F_node, target_node, weight="weight")
                        path_length = nx.dijkstra_path_length(G, F_node, target_node, weight="weight")
                    rounded_length = round(path_length, 2)
                    valid_paths.append((rounded_length, target_node, path))
                except nx.NetworkXNoPath:
                    continue

        valid_paths.sort(key=lambda x: x[0])
        neighborhoods[F_node_id] = set([coord_id_map.get(path[1], path[1]) for path in valid_paths])

        for path_length, target_node, path in valid_paths:
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
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0}
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




def plot_voronoi(vor):
    fig, ax = plt.subplots()


    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)


    # Customize the plot
    ax.set_title('Voronoi Diagram')
    plt.show()


def main():
    # Load road and hydrant data
    road_path = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\Road centerlines\roads_whole_region\centerlines_splited.shp"
    hydrants_path = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\hydrants\drive-download-20231202T013905Z-001\hydrants_whole_region.shp"
    roads = gpd.read_file(road_path).reset_index(drop=True)
    hydrants = gpd.read_file(hydrants_path).reset_index(drop=True)
    hydrant_shapefile_path = (
        r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\hydrants\drive-download-20231202T013905Z-001\hydrants_whole_region.shp"
    )
    hydrant_shapefile = gpd.read_file(hydrant_shapefile_path)
    print(hydrant_shapefile.columns)
    G, snapped_coords = create_network_snapped(roads, hydrants, search_radius=10)
    # Assign IDs to snapped coordinates
    coord_id_map = assign_ids_to_snapped_coords(snapped_coords, hydrant_shapefile)
    print(len(snapped_coords))
    print("assigned af")
    # Connect all disconnected components to the largest component
    #G = connect_components(G)
    # Find shortest paths using the IDs
    shortest_paths, neighborhoods =find_all_nearest_neighbors_voronoi(G, hydrants, roads, snapped_coords, coord_id_map)




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
    #visualize_network(G, roads, hydrants, snapped_coords, shortest_paths)
   
    # Save shortest paths to a new shapefile
    # Assuming paths_to_linestrings now returns a GeoDataFrame with all the necessary data
    #visualize_connected_components(G, roads, snapped_coords)
    # Now you don't need to create a new GeoDataFrame, as linestrings_gdf already is one
    output_path = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\outputs\mod_vor_paths.shp"
    output_path1 = r"C:\Users\funkt\OneDrive\Desktop\connected_network-20231115T233050Z-001\connected_network\outputs\mod_vor_neighbors.shp"
   
    save_neighborhoods_to_shapefile(neighborhoods, shortest_paths, coord_id_map, snapped_coords, output_path1)
    linestrings_gdf.to_file(output_path)
    lines = [LineString([source, target]) for source, target in G.edges()]
    gdf = gpd.GeoDataFrame(geometry=lines)






    vor = Voronoi(snapped_coords)
    plot_voronoi(vor)


# Save the GeoDataFrame as a shapefile
   # gdf.to_file(r"C:\Users\Thelonious\Desktop\connected_network-20231103T193127Z-001\connected_network\Gnetwork.shp")
   # print("All done")








if __name__ == "__main__":
    main()


