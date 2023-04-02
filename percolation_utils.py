# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import scipy.optimize as opt
import tqdm
import random
import time
import pickle
import tempfile
import os
import shutil
from scipy.optimize import curve_fit
import pickle
from datetime import datetime
from scipy.stats import linregress
#%%


def calculate_average_cluster_size(cluster_data, remove_giant_component=True):
    """
    Calculate the average cluster size for each occupation probability p. The average cluster size is defined as the
    sum of the squared sizes of all clusters divided by the total number of nodes in all clusters.

    Parameters:
    cluster_data (dict or list): A dictionary where the keys are the occupation probability values and the
                                 values are lists containing the size distribution of clusters, or a list
                                 containing the size distribution of clusters.
    remove_giant_component (bool): Whether to remove the largest cluster (giant component) from the calculation.
                                   Default is True.

    Returns:
    average_cluster_size (dict or float): If input is a dictionary, returns a dictionary where the keys are the
                                          occupation probability values and the values are the average cluster size
                                          for the corresponding probability. If input is a list, returns the
                                          average cluster size as a float.
    """

    if isinstance(cluster_data, dict):
        average_cluster_size = {}

        # Iterate through each occupation probability and its corresponding cluster size distribution
        for p in cluster_data:
            size_distribution = cluster_data[p]

            if remove_giant_component:
                largest_cluster = np.max(size_distribution)
                size_distribution = np.delete(size_distribution, np.argmax(size_distribution))

            total_nodes = sum(size_distribution)

            # If there are no nodes in the cluster, set the average cluster size to 0
            if total_nodes == 0:
                average_cluster_size[p] = 0
            else:
                # Calculate the average cluster size as the sum of squared sizes divided by the total number of nodes
                total_nodes_squared = sum(size_distribution**2)
                average_cluster_size[p] = total_nodes_squared / total_nodes

        return average_cluster_size

    elif isinstance(cluster_data, list):
        size_distribution = np.array(cluster_data)

        if remove_giant_component:
            largest_cluster = np.max(size_distribution)
            size_distribution = np.delete(size_distribution, np.argmax(size_distribution))

        total_nodes = sum(size_distribution)

        if total_nodes == 0:
            return 0
        else:
            total_nodes_squared = sum(size_distribution**2)
            return total_nodes_squared / total_nodes


def find_best_fit_beta(sorted_p_values, sorted_gcc_values, p_c, num_samples=20, min_points=5, max_points=None):
    """
    Find the best fit line for the given data points in a log-log plot to determine the critical exponent.

    Parameters:
    sorted_p_values (list): A sorted list of occupation probability values.
    sorted_gcc_values (list): A sorted list of the corresponding largest connected component (GCC) values.
    p_c (float): The critical occupation probability.
    num_samples (int, optional): The number of samples to consider for each iteration. Default is 20.
    min_points (int, optional): The minimum number of points to include in each sample. Default is 5.
    max_points (int, optional): The maximum number of points to include in each sample. Default is None, in which case
                                it is set to the length of sorted_p_values.

    Returns:
    best_start (int): The starting index of the best fit.
    best_n (int): The number of points in the best fit.
    best_slope (float): The slope of the best fit line.
    best_intercept (float): The intercept of the best fit line.
    best_r2 (float): The coefficient of determination (R²) of the best fit line.
    """
    if max_points is None:
        max_points = len(sorted_p_values)

    best_r2 = -np.inf
    best_slope = None
    best_intercept = None
    best_start = None
    best_n = None

    start_points = np.linspace(0, len(sorted_p_values) - min_points, num_samples, dtype=int)
    end_points = np.linspace(min_points, max_points, num_samples, dtype=int)

    # Iterate through all possible combinations of start and end points
    for start in start_points:
        for n in end_points:
            if start + n > len(sorted_p_values):
                continue

            filtered_p_values = sorted_p_values[start : start + n]
            filtered_gcc_values = sorted_gcc_values[start : start + n]

            # Perform linear regression on the log-log plot of the data
            slope, intercept, r_value, p_value, std_err = linregress(np.log(np.abs(filtered_p_values - p_c)), np.log(filtered_gcc_values))

            r2 = r_value**2

            # Update the best fit parameters if the current fit has a higher R² value
            if r2 > best_r2:
                best_r2 = r2
                best_slope = slope
                best_intercept = intercept
                best_start = start
                best_n = n

    return best_start, best_n, best_slope, best_intercept, best_r2


def save_output_data(L, GCC, GCC2, p, p_c, g_before_pc, closest_p, active_nodes_at_closest_p):
    # Create a folder with relevant information
    current_date = datetime.now().strftime("%Y%m%d")
    folder_name = f"results_L_{L}_date_{current_date}"
    os.makedirs(folder_name, exist_ok=True)

    # Save output data using pickle
    output_data = {
        'GCC': GCC,
        'GCC2': GCC2,
        'p': p,
        'p_c': p_c,
        'g_before_pc': g_before_pc,
        'closest_p': closest_p,
        'active_nodes_at_closest_p': active_nodes_at_closest_p
    }

    with open(f"{folder_name}/output_data.pickle", "wb") as f:
        pickle.dump(output_data, f)

    return folder_name

def load_output_data(folder_name):
    with open(f"{folder_name}/output_data.pickle", "rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data

def calculate_gcc_gcc2(G):
    """
    Calculate the size of the largest (GCC) and second largest (GCC2) connected components of a graph G.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    gcc (int): The size of the largest connected component.
    gcc2 (int): The size of the second largest connected component, or 0 if there is only one connected component.
    """
    # Find all connected components in G and sort them by size in descending order
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Get the size of the largest connected component
    gcc = len(connected_components[0])

    # Get the size of the second largest connected component if it exists, otherwise set it to 0
    if len(connected_components) > 1:
        gcc2 = len(connected_components[1])
    else:
        gcc2 = 0

    return gcc, gcc2

def simulate_percolation_2d_lattice_around_pc(L, g_before_pc, p_start, range_pc, num_points):
    """
    Simulate percolation on a 2D lattice around the critical occupation probability.

    Parameters:
    L (int): The size of the lattice.
    g_before_pc (networkx.Graph): The graph before percolation, in the form of a NetworkX Graph object.
    p_start (float): The starting value of the occupation probability.
    range_pc (float): The range around the critical occupation probability to simulate.
    num_points (int): The number of points to simulate in the given range.

    Returns:
    tuple: A tuple containing the following elements:
        - S (dict): A dictionary of the GCC sizes for each p value.
        - S2 (dict): A dictionary of the GCC^2 sizes for each p value.
        - p_values (numpy.ndarray): An array of the p values used in the simulation.
        - cluster_size_distributions (dict): A dictionary of the cluster size distributions for each p value.
        - average_cluster_sizes (dict): A dictionary of the average cluster sizes for each p value.
    """

    N = L * L
    G = g_before_pc.copy()

    # Calculate the p values around the critical point
    p_values = np.linspace(p_start, p_start - 2*range_pc, num_points)
    delta_p = np.abs(p_values[1] - p_values[0])

    # Randomize the order of nodes to remove
    nodes_to_remove_list = random.sample(G.nodes(), len(G.nodes()))

    # Prepare results storage as dictionaries
    S = {}
    S2 = {}
    cluster_size_distributions = {}
    average_cluster_sizes = {}

    # Calculate percolation for each p value
    for p in p_values:
        # Determine number of nodes to remove
        num_nodes_to_delete = int(delta_p * N)

        # Get list of nodes to remove
        nodes_to_kill_ids = nodes_to_remove_list[0:num_nodes_to_delete]

        # Remove nodes (and their edges) and update nodes_to_remove_list
        for node_tuple in nodes_to_kill_ids:
            G.remove_node(node_tuple)
            nodes_to_remove_list.pop(0)

        # Calculate GCC and GCC^2 for current graph state
        gcc, gcc2 = calculate_gcc_gcc2(G)
        S[p] = gcc
        S2[p] = gcc2

        # Calculate and store cluster size distribution for the current value of p
        connected_components = list(nx.connected_components(G))
        connected_components.sort(key=len, reverse=True)  # Sort connected components by size
        cluster_sizes = [len(cluster) for cluster in connected_components if len(cluster) > 1]

        if cluster_sizes:
            hist, _ = np.histogram(cluster_sizes, bins=np.arange(1, max(cluster_sizes) + 1))
        else:
            hist = np.array([])

        cluster_size_distributions[p] = hist

        # Calculate and store average cluster size for the current value of p
        average_cluster_sizes[p] = calculate_average_cluster_size(cluster_sizes, remove_giant_component=True)

    # Normalize GCC and GCC^2 by number of nodes
    S = {p: g / N for p, g in S.items()}
    S2 = {p: g / N for p, g in S2.items()}

    return S, S2, p_values, cluster_size_distributions, average_cluster_sizes



def simulate_percolation_2d_lattice_return_gcc_gcc2_pc(L, estimated_p_c=None, save_range=0.05):
    """
    Simulate bond percolation on a 2D square lattice, and return the size of the largest (GCC) and second largest
    (GCC2) connected components as a function of the occupation probability p, as well as the critical probability p_c.

    Parameters:
    L (int): The size of the lattice (L x L).
    estimated_p_c (float, optional): An estimate of the critical probability p_c. Default is None.
    save_range (float, optional): The range around the estimated p_c within which the graph states will be saved.
                                   Default is 0.05.

    Returns:
    GCC (list): The size of the largest connected component for each occupation probability p.
    GCC2 (list): The size of the second largest connected component for each occupation probability p.
    p (np.array): The occupation probability values.
    p_c (float): The critical probability p_c.
    g_before_pc (networkx.Graph): The graph state just before the occupation probability p_c is reached.
    closest_p (float): The occupation probability closest to p_c.
    active_nodes_at_closest_p (set): The set of active nodes at the occupation probability closest to p_c.
    """
    N = L * L
    G = nx.grid_2d_graph(L, L)
    
    # Prepare the occupation probability values
    delta_p = 0.01
    p_values = np.array([delta_p] * 99)
    
    # Get nodes list
    nodes = list(G)
    node_indices = set(range(len(nodes)))
    
    # Prepare nodes-to-remove vector 
    nodes_to_remove_list = random.sample(list(range(N)), N)
    
    # Prepare results storage
    GCC = {1: N}
    GCC2 = {1: 0}
    graph_list = {}
    active_nodes_at_closest_p = None
    deleted_nodes_list = {}
    
    # Calculate percolation
    for i, p in enumerate(p_values):
        # Determine the number of nodes to delete
        num_nodes_to_delete = int(delta_p * N)
       
        # Select the nodes to delete
        nodes_to_kill_ids = nodes_to_remove_list[0:num_nodes_to_delete]
        deleted_nodes_list[i * p] = nodes_to_kill_ids

        # Remove nodes and update the nodes_to_remove list
        edges_to_remove = set()
        for node_id in nodes_to_kill_ids:
            node = nodes[node_id]
            edges_to_remove.update(G.edges(node))
            G.remove_node(node)
            nodes_to_remove_list.pop(0)
        
        # Calculate GCC and GCC2
        gcc, gcc2 = calculate_gcc_gcc2(G)
        current_p = 1 - i * delta_p
        GCC[current_p] = gcc
        GCC2[current_p] = gcc2
        
        # Save graph state if it's within the save_range of the estimated p_c
        if estimated_p_c is not None and abs(current_p - estimated_p_c) <= save_range:
            graph_list[current_p] = G.copy()
                
    # Calculate the occupation probability values
    p = 1 - np.cumsum(p_values)
    p = np.concatenate(([1], p))
    
    # Find p_c
    GCC2_values_list = list(GCC2.values())
    pc_idx = np.argmax(GCC2_values_list)
    p_c = max(GCC2, key=GCC2.get)

    g_before_pc = None
    closest_p = None

    # Find the closest p to p_c and the corresponding graph state
    if graph_list:  # Only find the closest_p if the graph_list is not empty
        target_p = p_c + save_range
        closest_p = max([p for p in graph_list.keys() if p < target_p], default=None)
        if closest_p is not None:
            g_before_pc = graph_list[closest_p]

        # Determine the set of active nodes at the occupation probability closest to p_c
        all_deleted_nodes_up_to_closest_p = set()
        for key in deleted_nodes_list:
            if key > closest_p:
                all_deleted_nodes_up_to_closest_p |= set(deleted_nodes_list[key])

    
    # Normalize GCC and GCC2 values
    GCC = [g / N for g in GCC]
    GCC2 = [g / N for g in GCC2]

    return GCC, GCC2, p, p_c, g_before_pc
'''
def find_best_fit_gamma(sorted_p_values, sorted_avg_cluster_size_values, p_c, num_samples=100, min_points=40, max_points=None):
    threshold = 1e-8
    max_distance_from_pc = 0.02  # maximum distance from p_c allowed for data points
    
    if max_points is None:
        max_points = len(sorted_p_values) - min_points

    best_r2 = -np.inf
    best_slope = None
    best_intercept = None
    best_start = None
    best_n = None

    start_points = np.linspace(0, len(sorted_p_values) - min_points, num_samples, dtype=int)
    end_points = np.linspace(min_points, max_points, num_samples, dtype=int)

    for start in start_points:
        for n in end_points:
            if start + n > len(sorted_p_values):
                continue

            filtered_p_values = sorted_p_values[start : start + n]
            filtered_avg_cluster_size_values = sorted_avg_cluster_size_values[start : start + n]
            
            # Filter out values of p that are too far from p_c
            filtered_p_values = [p for p in filtered_p_values if abs(p - p_c) <= max_distance_from_pc]
            filtered_avg_cluster_size_values = [s for s, p in zip(filtered_avg_cluster_size_values, filtered_p_values)]

            slope, intercept, r_value, p_value, std_err = linregress(np.log([p_c - p for p in filtered_p_values if np.abs(p - p_c) > threshold]), np.log(filtered_avg_cluster_size_values))

            r2 = r_value**2

            if r2 > best_r2:
                best_r2 = r2
                best_slope = slope
                best_intercept = intercept
                best_start = start
                best_n = n

    return best_start, best_n, best_slope, best_intercept, best_r2
'''

def find_best_fit_gamma(sorted_p_values, sorted_avg_cluster_size_values, num_samples=100, min_points=40, max_points=None):
    
    if max_points is None:
        max_points = len(sorted_p_values) - min_points

    best_r2 = -np.inf
    best_slope = None
    best_intercept = None
    best_start = None
    best_n = None

    start_points = np.linspace(0, len(sorted_p_values) - min_points, num_samples, dtype=int)
    end_points = np.linspace(min_points, max_points, num_samples, dtype=int)

    for start in start_points:
        for n in end_points:
            if start + n > len(sorted_p_values):
                continue

            filtered_p_values = sorted_p_values[start : start + n]
            filtered_avg_cluster_size_values = sorted_avg_cluster_size_values[start : start + n]

            slope, intercept, r_value, p_value, std_err = linregress(np.log(filtered_p_values), np.log(filtered_avg_cluster_size_values))

            r2 = r_value**2

            if r2 > best_r2:
                best_r2 = r2
                best_slope = slope
                best_intercept = intercept
                best_start = start
                best_n = n

    return best_start, best_n, best_slope, best_intercept, best_r2

'''
'''
'''
def simulate_percolation_2d_lattice_around_pc(L, p_c, G, nodes_to_remove_list, small_range, num_steps):
    N = L * L
    start_p = p_c - small_range
    end_p = p_c + small_range
    p_values = np.linspace(start_p, end_p, num_steps)
    
    # Prepare results storage
    GCC = []
    GCC2 = []
    
    # Calculate percolation
    for p in p_values:
        num_nodes_to_delete = int(p * N) - (N - len(nodes_to_remove_list))
        nodes_to_kill_ids = nodes_to_remove_list[0:num_nodes_to_delete]
        # Kill nodes and update nodes_to_kill list
        for node_id in nodes_to_kill_ids:
            G.remove_node(nodes[node_id])
            nodes_to_remove_list.pop(0)

        gcc, gcc2 = calculate_gcc_gcc2(G)
        GCC.append(gcc)
        GCC2.append(gcc2)

    GCC = [g / N for g in GCC]
    GCC2 = [g / N for g in GCC2]

    return GCC, GCC2, p_values


def calculate_average_cluster_size(cluster_size_distributions, remove_giant_component=True):
    """
    Calculate the average cluster size for each occupation probability p. The average cluster size is defined as the
    sum of the squared sizes of all clusters divided by the total number of nodes in all clusters.

    Parameters:
    cluster_size_distributions (dict): A dictionary where the keys are the occupation probability values and the
                                      values are lists containing the size distribution of clusters.
    remove_giant_component (bool): If set to True, the giant component is removed from the cluster size distribution.

    Returns:
    average_cluster_size (dict): A dictionary where the keys are the occupation probability values and the values
                                 are the average cluster size for the corresponding probability.
    """
    
    def remove_zero_elements_from_arrays_in_dict(d):
        return {key: value[value != 0] for key, value in d.items()}

    cluster_size_distributions = remove_zero_elements_from_arrays_in_dict(cluster_size_distributions)

    average_cluster_size = {}

    # Iterate through each occupation probability and its corresponding cluster size distribution
    for p in cluster_size_distributions:
        size_distribution = cluster_size_distributions[p]
        total_nodes = sum(size_distribution)
        if remove_giant_component:
            size_distribution = size_distribution[1:]  # Exclude the giant component

        # If there are no nodes in the cluster, set the average cluster size to 0
        if total_nodes == 0:
            average_cluster_size[p] = 0
        else:
            # Calculate the average cluster size as the sum of squared sizes divided by the total number of nodes
            total_nodes_squared = sum([size**2 for size in size_distribution])
            average_cluster_size[p] = total_nodes_squared / total_nodes

    return average_cluster_size


'''
