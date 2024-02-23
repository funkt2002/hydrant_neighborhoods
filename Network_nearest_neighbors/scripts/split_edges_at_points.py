import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import split
from tqdm import tqdm
from shapely.geometry import GeometryCollection

def split_edges_at_points(edges, points, buffer_size=2):
    # Apply a buffer around each hydrant point to create a bounding box
    points['buffered_geometry'] = points['geometry'].buffer(buffer_size)

    # Creating a spatial index for the buffered points for efficient querying
    points_sindex = points.sindex

    # Initialize an empty list to store the split edges
    split_edges = []

    # Iterate over all edges
    for _, edge in tqdm(edges.iterrows(), total=edges.shape[0]):
        edge_geometry = edge.geometry
        edge_parts = [edge_geometry]
        print(edge_parts)
        # Find potential points for intersection using spatial index
        possible_matches_index = list(points_sindex.intersection(edge_geometry.bounds))
        possible_matches = points.iloc[possible_matches_index]

        # Iterate over possible matches
        for _, point in possible_matches.iterrows():
            buffered_point = point['buffered_geometry']

            new_edge_parts = []
            for part in edge_parts:
                if part.intersects(buffered_point):
                    # Split the part at the buffered point
                    split_result = split(part, buffered_point)

                    # Handle GeometryCollection result
                    if isinstance(split_result, GeometryCollection):
                        for geom in split_result.geoms:  # Use .geoms here
                            if isinstance(geom, LineString) and not geom.is_empty:
                                new_edge_parts.append(geom)
                    else:
                        new_edge_parts.extend(split_result)
                else:
                    new_edge_parts.append(part)
            edge_parts = new_edge_parts

        # Add the split parts to the list of split edges, preserving the attributes of the original edge
        for part in edge_parts:
            split_edge = edge.copy()
            split_edge.geometry = part
            split_edge['shape_len'] = part.length
            split_edges.append(split_edge)

    # Create a GeoDataFrame from the list of split edges
    split_edges_gdf = gpd.GeoDataFrame(split_edges, crs=edges.crs)

    return split_edges_gdf

def save_to_shapefile(gdf, filepath):
    gdf.to_file(filepath)


def main():
    edges = gpd.read_file(r'C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\road _centerlines\Clipped to region of analysis and splited at hydrant location points\roads_subset2_clipped.shp')
    points = gpd.read_file(r'C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\hydrants\HydrantswithLandUseandids\hydrants_whole_region.shp')  


    split_edges = split_edges_at_points(edges, points, buffer_size = 2)
    save_to_shapefile(split_edges, r'C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\road _centerlines\complete road network\subset_county_split.shp')

if __name__ == '__main__':
    main()