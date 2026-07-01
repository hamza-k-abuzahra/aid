import numpy as np
import re, os
from typing import Union
import random as rand
import pandas as pd
import json, math
from sklearn.manifold import MDS


def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def path_cost(nodes_coords):
    cost = 0
    for path_indx in range(len(nodes_coords) - 1):
        cost += euclidean_dist(nodes_coords[path_indx], nodes_coords[path_indx + 1])
    cost += euclidean_dist(nodes_coords[-1], nodes_coords[0])
    return cost


def natural_sort_key(s):
    """Key function for natural sorting of strings containing numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]

def walk_naturally_sorted(root_dir):
    """Generator that yields directories in natural sorted order"""
    for root, dirs, files in os.walk(root_dir):
        # Sort directories numerically before continuing traversal
        dirs[:] = sorted(dirs, key=natural_sort_key)  # In-place modification
        yield root, dirs, files


def write_csv(path: str, headers: list[str], rows: list[list[Union[str, float, int]]]):
    with open(path, "w") as f:
        # Write header
        f.write(",".join(headers) + "\n")

        # Write rows
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")

def parse_dumas_extended(path: str):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    total_dist = lines[0].split(" ")[-1]
    header_end = next(i for i, line in enumerate(lines) if line.startswith("CUST NO."))
    data_lines = lines[header_end + 1:]

    coords = []
    time_windows = []

    for line in data_lines:
        if line.startswith("999"):
            break
        parts = list(map(float, line.split()))
        _, x, y, _, ready, due, service = parts
        coords.append((x, y))
        time_windows.append((ready, due))

    coords = np.array(coords)
    time_windows = np.array(time_windows)
    n = len(coords)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = math.floor(np.linalg.norm(coords[i] - coords[j]))

    return coords, total_dist

def parse_tsplib_file(path: str, num_instances: int):
    instances_count = 2783
    positions_dataset = list(dict())
    with open(path, mode="r") as file:
        reader = pd.read_csv(file)
        reader = reader[reader["num_cities"] == 20]
        instances_count = len(reader)
        print(instances_count)
        cities = rand.sample(range(instances_count), num_instances)
        print(reader.iloc[0]["total_distance"])
        print(len(json.loads(reader.iloc[0]["city_coordinates"])))

        for i in cities:
            positions_dataset.append({"id": reader.iloc[i]["instance_id"], "coords": json.loads(reader.iloc[i]["city_coordinates"]), "bench_total_dist": reader.iloc[i]["total_distance"]})
    return positions_dataset
# instance_id, num_cities, city_coordinates, distance_matrix, best_route, total_distance


def read_distance_matrix(path: str):     
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        raise ValueError(f"File {path} is empty")
    
    # Try to parse as a square matrix
    matrix_data = []
    header_detected = False
    
    for i, line in enumerate(lines):
        # Try different delimiters: space, tab, comma
        if ',' in line:
            row = [x.strip() for x in line.split(',')]
        else:
            row = line.split()
        
        # Check if first row might be headers (contains non-numeric first element)
        if i == 0 and len(row) > 1:
            try:
                # Try converting first element to float
                float(row[0])
                # It's numeric, so no header
                header_detected = False
            except ValueError:
                # First element is not numeric, treat as header
                header_detected = True
                # Skip the header row
                continue
        
        # Convert row to floats
        try:
            numeric_row = [float(x) for x in row]
            matrix_data.append(numeric_row)
        except ValueError:
            # If conversion fails, this might be a header row - skip it
            continue
    
    # Convert to numpy array
    distance_matrix = np.array(matrix_data)
    
    # Check if matrix is square
    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError(f"Distance matrix must be square. Got shape: {distance_matrix.shape}")
    
    return distance_matrix

def validate_distance_matrix(distance_matrix):
    """
    Validate that the distance matrix has reasonable properties.
    """
    n = distance_matrix.shape[0]
    
    # Check diagonal is zero
    diag = np.diag(distance_matrix)
    if not np.allclose(diag, 0, atol=1e-6):
        print(f"Warning: Diagonal contains non-zero values: {diag}")
        # Fix by setting diagonal to zero
        np.fill_diagonal(distance_matrix, 0)
    
    # Check symmetry (or warn if asymmetric)
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-6):
        print("Warning: Distance matrix is not symmetric. Averaging to make symmetric.")
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Check for negative distances
    if np.any(distance_matrix < 0):
        print(f"Warning: Found negative distances. Minimum value: {np.min(distance_matrix)}")
        # Clip negative values to zero
        distance_matrix = np.maximum(distance_matrix, 0)
    
    return distance_matrix

def dist_matrix_to_coords(path: str, n_components: int = 2, random_state: int = 42): 
    dist_matrix = read_distance_matrix(path)

    validate_distance_matrix(dist_matrix)

    print(f"Loaded distance matrix of size {dist_matrix.shape[0]} x {dist_matrix.shape[1]}")

    mds = MDS(
        n_components=n_components,
        dissimilarity='precomputed',
        random_state=random_state,
        normalized_stress='auto',  # Better for comparing different solutions
        max_iter=300, 
        eps=1e-5
    )
    print("Computing coordinates using MDS...")
    coordinates = mds.fit_transform(dist_matrix)

    # Calculate stress (goodness of fit)
    stress = mds.stress_ / np.sum(dist_matrix**2)
    print(f"MDS stress: {stress:.4f} (lower is better, 0 = perfect fit)")

    return coordinates, dist_matrix   

def main():
    coords, distance_matrix = dist_matrix_to_coords("five_d.txt")
    # coords, total_dist = parse_dumas_extended("dumas/n20w120.001.txt")
    print(coords)
    print(distance_matrix)


if __name__ == "__main__": 
    main()