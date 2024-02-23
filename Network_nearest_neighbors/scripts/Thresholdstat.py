import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.affinity import translate
from shapely.geometry import Point, LineString, box
import networkx as nx
from tqdm import tqdm
import seaborn as sns




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
    for road in roads.geometry:
        if road.geom_type == 'LineString':
            for coord in road.coords:
                vertices.append(Point(round(coord[0], 3), round(coord[1], 3)))
        elif road.geom_type == 'MultiLineString':
            for line in road.geoms:  # Use .geoms here
                for coord in line.coords:
                    vertices.append(Point(round(coord[0], 3), round(coord[1], 3)))
    vertices_gdf = gpd.GeoDataFrame(geometry=vertices)

    # Iterate over all hydrants
    for _, hydrant in tqdm(hydrants.iterrows(), total=hydrants.shape[0]):
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
        if road.geometry.geom_type == 'LineString':
            for i in range(len(road.geometry.coords) - 1):
                start = (round(road.geometry.coords[i][0], 3), round(road.geometry.coords[i][1], 3))
                end = (round(road.geometry.coords[i+1][0], 3), round(road.geometry.coords[i+1][1], 3))
                G.add_edge(start, end, weight=LineString([start, end]).length)
        elif road.geometry.geom_type == 'MultiLineString':
            for line in road.geometry.geoms:
                for i in range(len(line.coords) - 1):
                    start = (round(line.coords[i][0], 3), round(line.coords[i][1], 3))
                    end = (round(line.coords[i+1][0], 3), round(line.coords[i+1][1], 3))
                    G.add_edge(start, end, weight=LineString([start, end]).length)
    # Identify intersections
    intersections = [node for node in G.nodes() if len(list(G.edges(node))) >= 3]
    all_distances = []
    for intersection in intersections:
        nearest_dist = min(Point(intersection).distance(Point(coord)) for coord in snapped_coords)
        all_distances.append(nearest_dist)
    print('mean distance from intersections to their nearest hydrant', np.mean(all_distances))
    print('Standard deviation of distance from intersections to their nearest hydrant:', np.std(all_distances))
    print('Median distance from intersections to their nearest hydrant:', np.median(all_distances))
    print('Minimum distance from intersections to their nearest hydrant:', min(all_distances))
    print('Maximum distance from intersections to their nearest hydrant:', max(all_distances))
    print('First quartile of distance from intersections to their nearest hydrant:', np.percentile(all_distances, 25))
    print('Third quartile of distance from intersections to their nearest hydrant:', np.percentile(all_distances, 75))
    print('Number of intersections:', len(intersections))



        # Process snapped hydrants to relocate them if they are within 200 feet of an intersection
    relocation_distance = 500 
    processed_nodes = set()
    moved_coords = []
    distances = []  # List to store the distances

    for i, node in enumerate(snapped_coords):
        if node in processed_nodes:
            continue

        nearest_intersection = None
        min_distance = float("inf")
        # Find the nearest intersection within the threshold
        for intersection in intersections:
            distance = Point(intersection).distance(Point(node))
            if distance < min_distance and distance < relocation_distance:
                min_distance = distance
                nearest_intersection = intersection
        
        # If a nearest intersection within the threshold was found, relocate the hydrant
        if nearest_intersection:
            snapped_coords[i] = nearest_intersection
            moved_coords.append(nearest_intersection)  # Add the moved node to moved_coords
            distances.append(min_distance)  # This distance is from the hydrant to its nearest intersection within the threshold
            processed_nodes.add(node)  # Mark this node as processed

    # Print the number of relocated nodes
    print(len(processed_nodes), "nodes relocated")
    # Print the number of hydrants relocated within each distance range
    print("Hydrants within 0-25 feet of intersection:", len([d for d in distances if 0 <= d < 25]))
    print("Hydrants within 25-50 feet of intersection:", len([d for d in distances if 25 <= d < 50]))
    print("Hydrants within 50-75 feet of intersection:", len([d for d in distances if 50 <= d < 75]))
    print("Hydrants within 75-100 feet of intersection:", len([d for d in distances if 75 <= d <= 100]))
    print("Hydrants within 100-125 feet of intersection:", len([d for d in distances if 100 <= d <= 125]))
    print("Hydrants within 125-150 feet of intersection:", len([d for d in distances if 125 <= d <= 150]))
    print("Hydrants within 150-175 feet of intersection:", len([d for d in distances if 150 <= d <= 175]))
    print("Hydrants within 175-200 feet of intersection:", len([d for d in distances if 175 <= d <= 200]))

    # Print the distance statistics
    if distances:
        print("Minimum distance:", min(distances))
        print("Maximum distance:", max(distances))
        print("Median distance:", np.median(distances))
        print("First quartile:", np.percentile(distances, 25))
        print("Third quartile:", np.percentile(distances, 75))
        print("Mean distance:", np.mean(distances))
        print("Standard deviation of relocation distances:", np.std(distances))
    else:
        print("No hydrants were moved within the threshold distance.")

    # Add updated snapped_coords as nodes
    for coord in snapped_coords:
        G.add_node(coord)

    print(len(intersections), "intersections found")
    print(len(snapped_coords))
    return G, snapped_coords, intersections, moved_coords, distances, all_distances


def plot_adjusted_relocation_distances(all_distances):
    if not all_distances:
        print("No distances to plot.")
        return
    
    # Create a figure for the boxplot without outliers and with arrow annotations
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=all_distances, showfliers=False, ax=ax)
    plt.title('Boxplot of Distances from Intersections to Nearest Hydrants')
    plt.xlabel('Distance (feet)')

    # Calculate the quartiles, min, and max
    quartiles = np.percentile(all_distances, [25, 50, 75])
    min_val = np.min(all_distances)
    max_val = np.max(all_distances)

    # Annotate the quartiles, min, and max
    for q in quartiles:
        ax.annotate(f'{q:.2f}', xy=(q, 0), xytext=(20, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center', color='black')
    
    ax.annotate(f'{min_val:.2f}', xy=(min_val, 0), xytext=(20, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center', color='black')

    ax.annotate(f'{max_val:.2f}', xy=(max_val, 0), xytext=(20, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center', color='black')

    plt.show() # Show the boxplot

    # Create a figure for the histogram with a kernel density estimate
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_distances, kde=True, ax=ax)
    plt.title('Distribution of Distances from Intersections to Nearest Hydrants')
    plt.xlabel('Distance (feet)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show() # Show the histogram


def plot_adjusted_relocation_distances_within_threshold(distances):
    if not distances:
        print("No distances to plot.")
        return
    
    # Create a figure for the boxplot without outliers and with arrow annotations
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=distances, showfliers=False, ax=ax)
    plt.title('Boxplot of Distances from Intersections to Nearest Hydrants within Threshold`')
    plt.xlabel('Distance (feet)')

    # Calculate the quartiles, min, and max
    quartiles = np.percentile(distances, [25, 50, 75])
    min_val = np.min(distances)
    max_val = np.max(distances)

    # Annotate the quartiles, min, and max
    for q in quartiles:
        ax.annotate(f'{q:.2f}', xy=(q, 0), xytext=(20, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center', color='black')
    
    ax.annotate(f'{min_val:.2f}', xy=(min_val, 0), xytext=(20, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center', color='black')

    ax.annotate(f'{max_val:.2f}', xy=(max_val, 0), xytext=(20, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'), ha='center', va='center', color='black')

    plt.show() # Show the boxplot

    # Create a figure for the histogram with a kernel density estimate
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(distances, kde=True, ax=ax)
    plt.title('Distribution of Distances from Intersections to Nearest Hydrants within Threshold')
    plt.xlabel('Distance (feet)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show() # Show the histogram


def calculate_percentages(distances):
    # Ensure distances is a NumPy array for easier calculations
    distances = np.array(distances)
    
    # Calculate the first and third quartiles
    first_quartile = np.percentile(distances, 25)
    third_quartile = np.percentile(distances, 75)
    
    # Find distances within the 25-50 ft range
    distances_25_50 = distances[(distances >= 25) & (distances < 50)]
    
    # Find distances greater than 70 ft
    distances_greater_than_100 = distances[(distances <100)]
    
    # Calculate the percentage of distances within the 25-50 ft range
    percentage_25_50 = (len(distances_25_50) / len(distances)) * 100
    
    # Calculate the percentage of distances greater than 70 ft
    percentage_greater_than_100 = (len(distances_greater_than_100) / len(distances)) * 100
    
    # Calculate the percentage of distances within the 1st quartile (less than the first quartile value)
    distances_within_first_quartile = distances[distances < first_quartile]
    percentage_within_first_quartile = (len(distances_within_first_quartile) / len(distances)) * 100
    
    # Calculate the percentage of distances within the 3rd quartile (greater than or equal to the third quartile value but less than the max distance)
    distances_within_third_quartile = distances[(distances >= third_quartile) & (distances <= max(distances))]
    percentage_within_third_quartile = (len(distances_within_third_quartile) / len(distances)) * 100
    
    print(f"Percentage of distances within the 25-50 ft range: {percentage_25_50:.2f}%")
    print(f"Percentage of distances greater than 100 ft: {percentage_greater_than_100:.2f}%")
    print(f"Percentage of distances within the 1st quartile: {percentage_within_first_quartile:.2f}%")
    print(f"Percentage of distances within the 3rd quartile: {percentage_within_third_quartile:.2f}%")









def main():
    # Load road and hydrant data
    road_path = r"C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\road _centerlines\complete road network\subset_county_split.shp"
    hydrants_path = r"C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\hydrants\hydrants_reprojec_to_roads.shp"
    roads = gpd.read_file(road_path).reset_index(drop=True)
    hydrants = gpd.read_file(hydrants_path).reset_index(drop=True)
    hydrant_shapefile_path = (
        r"C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\hydrants\HydrantswithLandUseandids\hydrants_whole_region.shp"
    )
    hydrant_shapefile = gpd.read_file(hydrant_shapefile_path)
    print(hydrant_shapefile.columns)
    G, snapped_coords, intersections, moved_coords, distances, all_distances = create_network_snapped(roads, hydrants, search_radius=5)
    plot_adjusted_relocation_distances(all_distances)
    plot_adjusted_relocation_distances_within_threshold(distances)
    calculate_percentages(distances)
    calculate_percentages(all_distances)

if __name__ == "__main__":
    main()