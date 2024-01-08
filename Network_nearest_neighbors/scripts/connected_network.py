import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.ops import nearest_points
import pandas as pd
import uuid

"""This is a script that takes a disconnected road network and connects the
 smaller components to the largest component using a buffer. This allows for 
 the creation of a single connected road network that can be used for network
 calculations."""
def connect_with_buffer_corrected(roads):
    # Create a NetworkX graph from the road data
    G = nx.Graph()
    for _, row in roads.iterrows():
        G.add_edge(row["FNODE"], row["TNODE"], geometry=row["geometry"])

    # Determine the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    # Extract edges' geometries for the largest connected component
    main_component_geoms = [
        data["geometry"]
        for u, v, data in G.edges(data=True)
        if u in largest_cc or v in largest_cc
    ]

    # For each of the smaller components, determine the closest point in that component to the main component
    new_rows = []
    for component in nx.connected_components(G):
        if component != largest_cc:
            component_geoms = [
                data["geometry"]
                for u, v, data in G.edges(data=True)
                if u in component or v in component
            ]
            # Creating a union of the component's geometries and then creating a buffer around it
            component_union = gpd.GeoSeries(component_geoms).unary_union
            component_buffer = component_union.buffer(2)

            # Finding the nearest geometry in the main component to the buffered component
            nearest_geom_in_main = nearest_points(
                component_buffer, gpd.GeoSeries(main_component_geoms).unary_union
            )[1]

            # Finding the nearest geometry in the component to the buffered component
            nearest_geom_in_component = nearest_points(
                component_buffer, component_union
            )[1]

            # Storing the new road data
            new_rows.append(
                {
                    "geometry": LineString(
                        [nearest_geom_in_component, nearest_geom_in_main]
                    ),
                    "FNODE": str(uuid.uuid4()),
                    "TNODE": str(uuid.uuid4()),
                }
            )

    # Use concat to add the new rows to the roads dataframe
    roads = pd.concat([roads, gpd.GeoDataFrame(new_rows)], ignore_index=True)

    return roads


# Load the original road dataset and create a deep copy
roads_original = gpd.read_file(
    r"C:\Users\chris\Desktop\Hydrant_Project_Data\Road centerlines-20230927T235600Z-001\Road centerlines\roads\sub_centerlines_splited.shp"
)
roads_copy = roads_original.copy(deep=True)

# Convert the coordinates of the road's start and end points to tuples
roads_copy['FNODE'] = roads_copy['geometry'].apply(lambda x: str(tuple(x.coords[0])))
roads_copy['TNODE'] = roads_copy['geometry'].apply(lambda x: str(tuple(x.coords[-1])))

connected_roads_refined_corrected = connect_with_buffer_corrected(roads_copy)
connected_roads_refined_corrected.to_file("connected_road_network.shp")