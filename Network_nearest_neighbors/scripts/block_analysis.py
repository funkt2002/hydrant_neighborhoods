from typing import List, Tuple, Dict
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import warnings
from tqdm import tqdm

# Define function to load shapefiles
def load_shapefiles(paths_shapefile: str, blocks_shapefile: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    paths_gdf = gpd.read_file(paths_shapefile)
    blocks_gdf = gpd.read_file(blocks_shapefile)
    return paths_gdf, blocks_gdf

# Define function to ensure CRS match
def ensure_or_set_crs_for_paths(paths_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # If paths_gdf has no CRS defined, set it to match blocks_gdf's CRS
    if paths_gdf.crs is None and blocks_gdf.crs is not None:
        paths_gdf.set_crs(blocks_gdf.crs, inplace=True)
    # If both have defined CRS but they don't match, convert paths_gdf CRS to match blocks_gdf
    elif paths_gdf.crs != blocks_gdf.crs:
        paths_gdf = paths_gdf.to_crs(blocks_gdf.crs)
    return paths_gdf

# Updated function to analyze paths within blocks, including blocks that contain shortest paths only
def analyze_paths_within_blocks(paths_gdf: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Create spatial index for paths
    paths_sindex = paths_gdf.sindex
    
    # Initialize a list to hold warnings
    warnings_list_optimized = []
    
    # Initialize a dictionary to hold block statistics
    block_stats_optimized = {}
    
    # Iterate over blocks using spatial index to find potential intersecting paths
    for _, block in tqdm(blocks_gdf.iterrows()):
        block_id = block['GEOID20']  # Adjust as needed
        possible_matches_index = list(paths_sindex.intersection(block.geometry.bounds))
        possible_matches = paths_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(block.geometry)]
        
        paths_within_block = [path.geometry.length for _, path in precise_matches.iterrows() if block.geometry.contains(path.geometry) or block.geometry.intersects(path.geometry)]
        
        # Only add block stats if there are paths within the block
        if paths_within_block:
            avg_length = sum(paths_within_block) / len(paths_within_block)
            block_stats_optimized[block_id] = {'average_length': avg_length, 'count': len(paths_within_block)}
        
        # Collect warnings for paths partially outside blocks
        for _, path in precise_matches.iterrows():
            if not (block.geometry.contains(path.geometry) or block.geometry.intersects(path.geometry)):
                warnings_list_optimized.append(f"Path with Tnode {path['source']} and Fnode {path['target']} is partially outside block {block_id}")
    
    # Convert block statistics to DataFrame
    statistics_df_optimized = pd.DataFrame.from_dict(block_stats_optimized, orient='index')
    return statistics_df_optimized, warnings_list_optimized

# Define function to print statistics and warnings
def print_statistics_and_warnings(statistics_df: pd.DataFrame, warnings_list: List[str]):
    print("Block Statistics (Average Length and Count of Paths):")
    print(statistics_df)
    print("\nWarnings for paths partially outside blocks:")
    for warning in warnings_list:
        print(warning)


# File paths (assuming all necessary files are correctly assembled into .shp format)
paths_shapefile_path = r"C:\Users\funkt\OneDrive\Desktop\Network_nearest_neighbors\network_nearest_neighbors\outputs\all_possible_paths_30ftintersectionthreshhold.shp"
blocks_shapefile_path = r"C:\Users\funkt\OneDrive\Desktop\block_data\tl_2020_06083_tabblock20.shp"

paths_gdf, blocks_gdf = load_shapefiles(paths_shapefile_path, blocks_shapefile_path)

# Ensure or set CRS for paths_gdf to match blocks_gdf, if necessary
paths_gdf = ensure_or_set_crs_for_paths(paths_gdf, blocks_gdf)

# Proceed with analyzing paths within blocks and printing statistics and warnings
statistics_df, warnings_list = analyze_paths_within_blocks(paths_gdf, blocks_gdf)
print_statistics_and_warnings(statistics_df, warnings_list)