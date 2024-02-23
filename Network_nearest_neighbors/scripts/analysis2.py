import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box
import networkx as nx
from tqdm import tqdm
import seaborn as sns


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
           for line in road.geoms:
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
   intersections = [node for node in G.nodes() if len(list(G.edges(node))) == 3]
   nearest_distances = []
   for intersection in intersections:
       nearest_distance = min(Point(intersection).distance(Point(coord)) for coord in snapped_coords)
       nearest_distances.append(nearest_distance)


   if nearest_distances:
       print("Mean distance from intersections to their closest hydrant:", np.mean(nearest_distances).round(2))
   else:
       print("No distances found.")


   # Calculate distances from intersections to nearest hydrant within a given threshold
   threshold_distance = 250  # Define your threshold distance here
   distances_within_threshold = []


   # Find the nearest hydrant to each intersection and filter by threshold
   for intersection in intersections:
       nearest_distance = float('inf')
       nearest_coord = None
       for coord in snapped_coords:
           distance = Point(intersection).distance(Point(coord))
           if distance < nearest_distance:
               nearest_distance = distance
               nearest_coord = coord
      
       # Check if the nearest distance is within the threshold and add it to the list
       if nearest_distance <= threshold_distance:
           distances_within_threshold.append(nearest_distance)


   # Calculate and print statistics for distances within the threshold
   if distances_within_threshold:
       mean_distance = np.mean(distances_within_threshold)
       median_distance = np.median(distances_within_threshold)
       first_quartile = np.percentile(distances_within_threshold, 25)
       third_quartile = np.percentile(distances_within_threshold, 75)
      
       print(f"Mean distance for nearest hydrants within {threshold_distance} units: {mean_distance}")
       print(f"Median distance: {median_distance}")
       print(f"First quartile: {first_quartile}")
       print(f"Third quartile: {third_quartile}")
       print(f"Standard deviation: {np.std(distances_within_threshold)}")
       distances_within_100_feet = [d for d in distances_within_threshold if d <= 100]
       distances_within_200_feet = [d for d in distances_within_threshold if d <= 200]

       print(f"The percentage of intersections that have a hydrant within 100 feet is: {len(distances_within_100_feet)/len(intersections)*100}%")
       print(f"The percentage of intersections that have a hydrant within 200 feet is: {len(distances_within_200_feet)/len(intersections)*100}%")
       print(f"the mode bin is: {sns.histplot(distances_within_threshold, kde=True, binwidth=50).mode()}")
       
   else:
       print(f"No hydrants found within {threshold_distance} units of an intersection.")


   return G, snapped_coords, intersections, distances_within_threshold




def plot_adjusted_relocation_distances(distances):
   if not distances:
       print("No distances to plot.")
       return


   plt.figure(figsize=(10, 6))
   plt.subplot(2, 1, 1)
   sns.boxplot(x=distances)
   plt.title('Boxplot of Distances from Intersections to Nearest Hydrants')


   plt.subplot(2, 1, 2)
   sns.histplot(distances, kde=True)
   plt.title('Distribution of Distances from Intersections to Nearest Hydrants')
   plt.xlabel('Distance (feet)')
   plt.ylabel('Frequency')


   plt.tight_layout()
   plt.show()


def main():
   # Paths to your data files
   road_path = r"C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\road _centerlines\complete road network\subset_county_split.shp"
   hydrants_path = r"C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\hydrants\HydrantswithLandUseandids\hydrants_whole_region.shp"
  
   # Load road and hydrant data
   roads = gpd.read_file(road_path).reset_index(drop=True)
   hydrants = gpd.read_file(hydrants_path).reset_index(drop=True)


   # Define the search radius for finding vertices near hydrants
   search_radius = 5  # Adjust as necessary


   # Create the network and calculate distances
   G, snapped_coords, intersections, distances_within_threshold = create_network_snapped(roads, hydrants, search_radius)
  
   # Plot the distances
   if distances_within_threshold:
       plot_adjusted_relocation_distances(distances_within_threshold)
   else:
       print("No distances found within the specified threshold.")


if __name__ == "__main__":
   main()







