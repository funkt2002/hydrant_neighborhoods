# hydrant_neighborhoods
This project aims to determine neighborhoods of hydrants along a road network as well as consequent shortest paths using multiple different algorithms

All included scripts read shapefiles of Santa Barbara Road and hydrant data, and uses shapely and NetworkX to perform network analysis; creating different algorithms for identifying neighboring hydrants to a given source hydrant as well as shortest paths to them. 

There are three key scripts/algorithms:

1. All_possible_paths will define neighbors as any hydrant that is reachable along the network without traversing through another hydrant. There is no edge # constraint or network distance constraint.(Uses traversal until hydrant is met logic)

2. K_nearest_neighbors will identify K nearest nearest neighbors of a given source node to create a neighborhood with a few other contstraints. Firstly, a nearest neighbor cannot traverse through an already closer hydrant node along the same direction, as it would be redundant. Second, nearest neighbors must be found in either direction along a road from a given hydrant, even if closer ones lie in the other direction. (Uses djikstra to defind shortest paths)

3. The final script voronoi_modified creates voronoi polygons around each hydrant node and will define neighboring hydrant nodes by determining what other hydrant nodes are in neighboring polygons. Along with this logic, there is a constraint that a shortest path to a certain neighboring hydrant cannot traverse through another hydrant. (Uses djikstra to defind shortest paths)
