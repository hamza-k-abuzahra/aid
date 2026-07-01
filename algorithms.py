from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from shapely.geometry import Polygon  
from scipy.spatial import Voronoi
from deap import base, creator, tools, algorithms
import cv2

from utils import *
import random


def nearest_neighbor_tsp(start, nodes, positions):
    unvisited = set(nodes)
    path = [start]
    while unvisited:
        current = path[-1]
        # nearest neighbor heuristic 
        next_node = min(unvisited, key=lambda n: euclidean_dist(positions[current], positions[n]))
        path.append(next_node)
        unvisited.remove(next_node)
    path.pop(0)
    path.append(start)
    return path


def nearest_neighbor_mtsp(start_loc, nodes, positions):
    unvisited = set(nodes)
    start = min(unvisited, key=lambda n: euclidean_dist(start_loc, positions[n]))
    path = [start]
    unvisited.remove(start)
    while unvisited:
        current = path[-1]
        # nearest neighbor heuristic 
        next_node = min(unvisited, key=lambda n: euclidean_dist(positions[current], positions[n]))
        path.append(next_node)
        unvisited.remove(next_node)
    return path


def tsp_fitness(individual, waypoints):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += euclidean_dist(waypoints[individual[i]], waypoints[individual[i + 1]])
    return total_distance,


def compute_MST(positions: np.ndarray):
    dist_matrix = squareform(pdist(positions))
    mst = csgraph.minimum_spanning_tree(dist_matrix)
    return mst


def longest_edge_MST(mst):
    longest_edge = mst.max()
    
    # is_sparse = longest_edge > self.n_agents
    return longest_edge


def minimum_enclosing_circle(positions: np.ndarray):
    center, radius = cv2.minEnclosingCircle(positions)

    return center, radius


def kmeans_mtsp(leaders, targets, positions):
    routes = [[] for _ in leaders]

    num_leaders = len(leaders)
    num_targets = len(targets)

    if num_leaders >= num_targets:
        # More leaders than targets → assign one target per leader
        for i, target in enumerate(targets):
            routes[i] = [target.id]
        return routes


    waypoints = np.array([t.initial_pos for t in targets])
    kmeans = KMeans(n_clusters=num_leaders, random_state=0, n_init=10)
    labels = kmeans.fit_predict(waypoints)

    clusters = {i: [] for i in range(num_leaders)}
    for i, label in enumerate(labels):
        clusters[label].append(targets[i].id)
    


    start_pos = leaders[0].curr_pos  # same for all leaders
    for i, cluster_targets in clusters.items():
        if not cluster_targets:
            continue

        path = nearest_neighbor_mtsp(start_pos, cluster_targets, positions)

        routes[i] = path
    return routes


##########################################################################################################
###############                                 NOT USED                                  ################
##########################################################################################################
#### NOTE NOT USED
def genetic_alg_impl(waypoints, population_size=500, generations=350):
    """Solves TSP using Genetic Algorithm."""
    num_cities = len(waypoints)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_cities), num_cities)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", tsp_fitness, waypoints=waypoints)

    pop = toolbox.population(n=population_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.85, mutpb=0.3, ngen=generations, 
                        stats=None, halloffame=None, verbose=False)
    best_individual = tools.selBest(pop, k=1)[0]
    
    return best_individual

#### NOTE NOT USED
def genetic_algorithm_tsp(start, nodes, positions):
    waypoints = np.array([positions[p] for p in nodes])   
    best_route = genetic_alg_impl(waypoints, population_size=150, generations=550)
    best_path = [nodes[i] for i in best_route] 
    best_path.append(start) 

    # # get value of the path
    # waypoints = np.array([positions[p] for p in best_path])   
    # print(f"initial path cost: {path_cost(waypoints)}")

    return best_path

#### NOTE NOT USED
def voronoi_density(positions):
    vor = Voronoi(positions)
    densities = []

    for region in vor.regions:
        # Skip infinite regions and empty regions
        if not region or -1 in region:
            continue
            
        polygon = Polygon([vor.vertices[i] for i in region])
        
        if polygon.area > 0:
            densities.append(1 / polygon.area)

    return np.mean(densities) if densities else 0.0


#### NOTE NOT USED
def knn_density(positions, k=3):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)  # +1 to exclude self
    distances, _ = nbrs.kneighbors(positions)
    return 1 / np.mean(distances[:, 1:])