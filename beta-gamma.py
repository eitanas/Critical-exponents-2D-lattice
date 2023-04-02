#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:25:43 2023

@author: Eitan
"""
#%%
from percolation_utils import *
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

# Known critical exponents for 2D percolation
known_gamma = 43 / 18
known_beta = 5/36
#%%



def simulate_percolation_2d_lattice_around_pc(L, g_before_pc, p_start, range_pc, num_points):
    N = L * L
    G = g_before_pc.copy()

    # Calculate the p values around the critical point
    p_values = np.linspace(p_start, p_start - 2*range_pc, num_points)
    delta_p = np.abs(p_values[1] - p_values[0])
    
    nodes_to_remove_list = random.sample(G.nodes(), len(G.nodes()))
    
    # Prepare results storage as dictionaries
    S = {}
    S2 = {}
    cluster_size_distributions = {}
    average_cluster_sizes = {}
    
    # Calculate percolation
    for p in p_values:
        #p=p_values[0]
        num_nodes_to_delete = int(delta_p * N)
        nodes_to_kill_ids = nodes_to_remove_list[0:num_nodes_to_delete]
        
        # Remove nodes (and their edges) and update nodes_to_remove_list
        for node_tuple in nodes_to_kill_ids:
            G.remove_node(node_tuple)
            nodes_to_remove_list.pop(0)
        
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
        
    
        average_cluster_sizes[p] = calculate_average_cluster_size(cluster_sizes, remove_giant_component=True)

    S = {p: g / N for p, g in S.items()}
    S2 = {p: g / N for p, g in S2.items()}
    
    return S, S2, p_values, cluster_size_distributions, average_cluster_sizes

#%%  Percolation
L = 500  # Size of the 2D lattice
estimated_p_c = 0.59  # Optional: Estimated critical point
save_range = 0.06  # Optional: Save graphs within this range from the estimated critical point

RUN_SIM=1
if RUN_SIM:
    # Call the function with the provided arguments
    GCC, GCC2, p, p_c, g_before_pc, closest_p, active_nodes_at_closest_p,cluster_size_distributions = simulate_percolation_2d_lattice_return_gcc_gcc2_pc(L, estimated_p_c, save_range)
    print(f"p_c = {p_c}")
    print(closest_p)
    
    gcc_before_size, gcc2 = calculate_gcc_gcc2(g_before_pc)
    print(f"gcc at {closest_p} = {gcc_before_size}")
    # Save output data 
    
    folder_name = save_output_data(L, GCC, GCC2, p, p_c, g_before_pc, closest_p, active_nodes_at_closest_p)

LOAD_PREVIUOS_RESULTS=0
if LOAD_PREVIUOS_RESULTS:
    loaded_data = load_output_data(folder_name)
#%% Calculate high resolution
# Define the parameters for the second function
range_pc = save_range
num_points = 200

# Call the second function with the provided arguments                                                                    L, g_before_pc, active_nodes_at_closest_p, p_start, range_pc, num_points   
GCC_around_pc, GCC2_around_pc, p_values_around_pc, cluster_size_distributions, average_cluster_sizes = simulate_percolation_2d_lattice_around_pc(L,  g_before_pc,                             closest_p, range_pc, num_points )
# Extract p values and corresponding average cluster sizes
p_values = list(average_cluster_sizes.keys())
avg_cluster_sizes = list(average_cluster_sizes.values())

# Gamma fit
threshold = 1e-10
# Extract p values and corresponding average cluster sizes
filtered_p_values = [p for p in p_values if p < p_c and p_c - p > threshold]
filtered_avg_cluster_sizes = [s for s, p in zip(avg_cluster_sizes, p_values) if p < p_c and p_c - p > threshold]

best_start, best_n, best_slope, best_intercept, best_r2 = find_best_fit_gamma(filtered_p_values, filtered_avg_cluster_sizes, p_c)
gamma = - best_slope

#%%
# Plot the average cluster size as a function of p values
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(p_values, avg_cluster_sizes, 'o-', markersize=5, linewidth=2)
plt.plot(filtered_p_values[best_start: best_start + best_n], fit, '--', linewidth=2, label=f'Best fit: γ={gamma:.2f}, R^2={best_r2:.3f}')

# Add a vertical dashed line at p_c
plt.axvline(x=p_c, linestyle='--', color='r', label=f'p_c = {p_c}')

# Add text box with gamma values
plt.text(0.05, 0.15, f'Known γ: {known_gamma:.3f}\nCalculated γ: {gamma:.3f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

plt.xlabel("Occupation probability (p)")
plt.ylabel("Average cluster size")
plt.title("Average cluster size (without giant component) vs. Occupation probability")
plt.grid()
plt.legend()
plt.show()

#%%
#%%
#%%
#%%
#%%
#%%

#%% Beta calculation
# Extract the n values closer to p_c
sorted_p_values = np.array(list(GCC_around_pc.keys()))
sorted_gcc_values = np.array(list(GCC_around_pc.values()))
sorted_avg_cluster_size_values = np.array(list( average_cluster_size.values()))
best_start, best_n, best_slope, best_intercept, best_r2 = find_best_fit_beta(sorted_p_values, sorted_gcc_values, p_c)
# Number of values to use for the regression

# Extract the n values closer to p_c
filtered_p_values = sorted_p_values[best_start : best_start + best_n]
filtered_gcc_values = sorted_gcc_values[best_start : best_start + best_n]
filtered_avg_cluster_size_values = sorted_avg_cluster_size_values[best_start : best_start + best_n]
# Perform linear regression on log-transformed data
slope, intercept, r_value, p_value, std_err = linregress(np.log(np.abs(filtered_p_values - p_c)), np.log(filtered_gcc_values))

# The slope of the linear regression is an estimate of the beta exponent
beta = slope

print("Estimated beta exponent:", beta)

#%% plotting beta and gamma

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the log-log plot with the best linear regression line for beta
axes[0].loglog(np.abs(filtered_p_values - p_c), filtered_gcc_values, 'bo', color='darkorange', label='Data')
axes[0].loglog(np.abs(filtered_p_values - p_c), np.exp(best_intercept) * (np.abs(filtered_p_values - p_c))**best_slope, 'r', color='blue', label=f'Fit (β={best_slope:.3f}, R²={best_r2:.3f})')
axes[0].set_xlabel(r'$|p - p_c|$')
axes[0].set_ylabel(r'$P_\infty$')
axes[0].legend()
axes[0].text(0.01, 0.01, f'Known β: {known_beta:.3f}\nCalculated β: {beta:.3f}', transform=axes[0].transAxes, fontsize=12)

# Second subplot (Gamma exponent calculation)
axes[1].loglog(np.abs(np.array(filtered_p_values) - p_c), filtered_avg_cluster_size_values, 'bo', color='darkorange', label='Data')
axes[1].loglog(np.abs(np.array(filtered_p_values) - p_c), np.exp(gamma_intercept) * (np.abs(np.array(filtered_p_values_gamma) - p_c))**gamma_slope, 'r', color='blue', label=f'Best Fit (γ={estimated_gamma:.3f}, R²={gamma_r_value**2:.3f})')
axes[1].set_xlabel('$|p-p_c|$', fontsize=14)
axes[1].set_ylabel('Average Cluster Size', fontsize=14)
axes[1].legend(fontsize=12)
axes[1].set_title(f'Gamma Exponent Calculation (r-squared: {gamma_r_value**2:.5f})', fontsize=16)

# Show the figure
plt.tight_layout()
plt.show()















#%%
#%%
#%%
#%%
#%%
'''
import PySimpleGUI as sg

layout = [
    [sg.Text("Lattice Size (L):"), sg.Input(key="L")],
    [sg.Text("Number of Simulations:"), sg.Input(key="num_simulations")],
    [sg.Button("Submit"), sg.Button("Cancel")],
]

window = sg.Window("Input", layout)

while True:
    event, values = window.read()
    if event == "Submit":
        try:
            L = int(values["L"])
            num_simulations = int(values["num_simulations"])
            break
        except ValueError:
            sg.popup_error("Please enter valid integers")
    elif event == sg.WIN_CLOSED or event == "Cancel":
        break

window.close()

if event == "Submit":
    print(f"Lattice Size (L): {L}")
    print(f"Number of Simulations: {num_simulations}")
'''

#%%
def simulate_percolation_2d_lattice_return_gcc_gcc2_pc(L):
    N = L * L
    p_values = np.array([0.01] * 99)

    G = nx.grid_2d_graph(L, L)
    
    # Get nodes list
    nodes = list(G)
    # Prepare nodes-to-remove vector 
    nodes_to_remove_list = random.sample(list(range(N)), N)
    # Prepare results storage
    GCC = [N]
    GCC2 = [0]
    
    # Calculate percolation
    for p in p_values:
        num_nodes_to_delete = int(p * N)
        nodes_to_kill_ids = nodes_to_remove_list[0:num_nodes_to_delete]
        # Kill nodes and update nodes_to_kill list
        for node_id in nodes_to_kill_ids:
            G.remove_node(nodes[node_id])
            nodes_to_remove_list.pop(0)

        gcc, gcc2 = calculate_gcc_gcc2(G)
        GCC.append(gcc)
        GCC2.append(gcc2)

    p = 1 - np.cumsum(p_values)
    p = np.concatenate(([1], p))
    pc_idx = np.argmax(GCC2)
    p_c = p[pc_idx]

    GCC = [g / N for g in GCC]
    GCC2 = [g / N for g in GCC2]

    return GCC, GCC2, p, p_c

L = 100
GCC, GCC2, p, p_c, G, nodes_to_remove_list = simulate_percolation_2d_lattice_return_gcc_gcc2_pc(L)

small_range = 0.05
num_steps = 100
GCC_around_pc, GCC2_around_pc, p_values_around_pc = simulate_percolation_2d_lattice_around_pc(L, p_c, G, nodes_to


#%%

def simulate_percolation_2d_lattice_return_gcc_gcc2_pc(L):
    N = L * L
    p_values = np.array([0.01] * 99)

    G = nx.grid_2d_graph(L, L)
    
    # Get nodes list
    nodes = list(G)
    # Prepare nodes-to-remove vector 
    nodes_to_remove_list = random.sample(list(range(N)), N)
    # Prepare results storage
    GCC = [N]
    GCC2 = [0]
    
    # Calculate percolation
    for p in p_values:
        num_nodes_to_delete = int(p * N)
        nodes_to_kill_ids = nodes_to_remove_list[0:num_nodes_to_delete]
        # Kill nodes and update nodes_to_kill list
        for node_id in nodes_to_kill_ids:
            G.remove_node(nodes[node_id])
            nodes_to_remove_list.pop(0)

        gcc, gcc2 = calculate_gcc_gcc2(G)
        GCC.append(gcc)
        GCC2.append(gcc2)

    p = 1 - np.cumsum(p_values)
    p = np.concatenate(([1], p))
    pc_idx = np.argmax(GCC2)
    p_c = p[pc_idx]

    GCC = [g / N for g in GCC]
    GCC2 = [g / N for g in GCC2]

    return GCC, GCC2, p, p_c, G.copy(), nodes_to_remove_list

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

#%%
#%%
#%%
#%% test one simulation :
    
start_time = time.time()
L = 100  # Size of the 2D lattice
estimated_p_c = 0.59  # Optional: Estimated critical point
save_range = 0.05  # Optional: Save graphs within this range from the estimated critical point

# Call the function with the provided arguments
GCC, GCC2, p, p_c, g_pc, cluster_size_distributions = simulate_percolation_2d_lattice_return_gcc_gcc2_pc_gammaexp(L, estimated_p_c, save_range)

end_time = time.time()

running_time = end_time - start_time
print(f"Running time of simulation: {running_time:.3f} seconds")
#%%

# Calculate the beta exponent
p_min, p_max = p_c - 0.1, p_c
selected_GCC = [gcc for i, gcc in enumerate(GCC) if p_min <= p[i] <= p_max]
selected_p = [p_val for p_val in p if p_min <= p_val <= p_max]

def power_law(x, a, b):
    return a * x**b

p_diff_beta = np.abs(np.array(selected_p) - p_c)
popt_beta, _ = curve_fit(power_law, p_diff_beta, selected_GCC)
fitted_GCC = power_law(p_diff_beta, *popt_beta)
beta = popt_beta[1]

# Calculate the gamma exponent
p_min, p_max = p_c + 0.015, p_c + 0.35
selected_distributions = {p: cluster_size_distributions[p] for p in cluster_size_distributions if p_min <= p <= p_max}
fluctuations = {}

for p, distribution in selected_distributions.items():
    cluster_sizes = np.arange(1, len(distribution) + 1)
    sum_cluster_sizes_squared = np.sum(cluster_sizes**2 * distribution)
    total_num_nodes = np.sum(cluster_sizes * distribution)
    quantity = sum_cluster_sizes_squared / total_num_nodes
    fluctuations[p] = quantity

p_values = np.array(list(fluctuations.keys()))
fluctuations = N * np.array(list(fluctuations.values()))

popt_gamma, _ = curve_fit(power_law, p_values, fluctuations)
fitted_fluctuations = power_law(p_values, *popt_gamma)
gamma = popt_gamma[1]

# Plot the beta and gamma exponents in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Beta exponent plot
ax1.plot(p_diff_beta, selected_GCC, 'bo', label='Data', color='blue')
ax1.plot(p_diff_beta, fitted_GCC, 'r-', label=f'Fit: a={popt_beta[0]:.2e}, β={beta:.2f}', color='red')
ax1.set_xlabel(r'$|p - p_c|$')
ax1.set_ylabel('GCC')
ax1.legend()
ax1.set_title(f'Beta Exponent: {beta:.2f}')

# Gamma exponent plot
p_diff_gamma = p_values - p_c
valid_indices = np.where(p_diff_gamma > 1e-6)
filtered_p_diff = p_diff_gamma[valid_indices]
filtered_fluctuations = fluctuations[valid_indices]
filtered_fitted_fluctuations = fitted_fluctuations[valid_indices]

ax2.loglog(filtered_p_diff, filtered_fluctuations, 'bo', label='Data', color='blue')
ax2.loglog(filtered_p_diff, filtered_fitted_fluctuations,



known_gamma = 43 / 18  # Replace this with the known gamma value
analytical_p_c = 0.5927
analytical_p_c = p_c
# Adjust the range of p values for the fitting
p_min, p_max = analytical_p_c + 0.0005, analytical_p_c + 0.01
# Filter the cluster size distributions for the specified range of p values
selected_distributions = {
    p: cluster_size_distributions[p]
    for p in cluster_size_distributions
    if p_min <= p <= p_max
}
# Initialize an empty dictionary to store the calculated quantities for each p value
fluctuations = {}

# Calculate the desired quantity for each selected value of p
for p, distribution in selected_distributions.items():
    cluster_sizes = np.arange(1, len(distribution) + 1)
    sum_cluster_sizes_squared = np.sum(cluster_sizes**2 * distribution)
    total_num_nodes = np.sum(cluster_sizes * distribution)
    quantity = sum_cluster_sizes_squared / total_num_nodes
    fluctuations[p] = quantity

# Define the power-law function for curve fitting
def power_law(x, a, b):
    return a * x**b

# Convert the dictionaries to lists for easier plotting and fitting
p_values = np.array(list(fluctuations.keys()))
fluctuations = N*np.array(list(fluctuations.values()))

# Perform a power-law fit to the data
popt, _ = curve_fit(power_law, p_values, fluctuations)

# Calculate the fitted values
fitted_fluctuations = power_law(p_values, *popt)

# Update the x-axis to display (p - p_c)
p_diff = p_values - p_c

threshold = 1e-6  # Adjust this threshold value as needed

# Filter out the data points where p_diff is below the threshold
valid_indices = np.where(p_diff > threshold)
filtered_p_diff = p_diff[valid_indices]
filtered_fluctuations = fluctuations[valid_indices]
filtered_fitted_fluctuations = fitted_fluctuations[valid_indices]

# Plot the data and the fit on a log-log scale
plt.loglog(filtered_p_diff, filtered_fluctuations, 'bo', label='Data', color='blue')
plt.loglog(filtered_p_diff, filtered_fitted_fluctuations, 'r-', label=f'Fit: a={popt[0]:.2e}, γ={popt[1]:.2f}', color='red')

plt.xlabel(r'$p - p_c$')
plt.ylabel('Fluctuations')

# Add the calculated and known gamma values to the plot
bbox_props = dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1)
plt.annotate(f'Calculated γ: {popt[1]:.2f}', xy=(0.7, 0.2), xycoords='axes fraction', bbox=bbox_props)
plt.annotate(f'Known γ: {known_gamma:.2f}', xy=(0.7, 0.1), xycoords='axes fraction', bbox=bbox_props)

plt.tight_layout()
plt.show()

#%%
#%%
#%%
#%%

#%% After I found pc and I have g_pc I want to test how gcc is rising 
p_step = 0.01

def get_giant_cluster(graph):
    return max(nx.connected_components(graph), key=len)

def get_connected_components(graph):
    return list(nx.connected_components(graph))

def get_unconnected_nodes(graph, total_nodes):
    connected_nodes = set(n for cc in nx.connected_components(graph) for n in cc)
    return list(set(total_nodes) - connected_nodes)

g_pc_=g_pc.copy()
G=nx.grid_2d_graph(L,L)
gcc_size_at_pc = len(g_pc)/N
nodes=list(G)
gccMass=0

connected_components = get_connected_components(g_pc_)
unconnected_nodes = get_unconnected_nodes(g_pc_, nodes)
# Initialize a list to store the giant component size at each step
giant_component_sizes = []

while unconnected_nodes:
    num_nodes_to_connect = int(p_step * N)
    nodes_to_connect = random.sample(unconnected_nodes, min(num_nodes_to_connect, len(unconnected_nodes)))

    for node in nodes_to_connect:
        # Choose a random connected component
        target_component = random.choice(connected_components)
        target_node = random.choice(list(target_component))
        g_pc_.add_edge(node, target_node)

        # Update the connected components and unconnected nodes
        connected_components = get_connected_components(g_pc_)
        unconnected_nodes = get_unconnected_nodes(g_pc_, nodes)

    # Calculate and print the new size of the giant cluster
    giant_cluster = max(connected_components, key=len)
    gcc_size = len(giant_cluster)
    print("Giant cluster size:", gcc_size)
    # Add the current giant component size to the list
    giant_component_sizes.append(gcc_size)
    
print("Giant component sizes at each step:", giant_component_sizes)
    
    #%%
    
    
    
    #%%
    
    #%%
    #%%

for i in tqdm( range(100)):   
    rho_step = 0.001
    num_nodes_to_add = int(rho_step*N)
    
    
    for j in range(num_nodes_to_add): 
        # get a random node
        random_node = random.choice(list(nodes))          
        # test if this node belongs to the gcc
        if random_node not in g_pc_:              
            gcc_node = random.choice(list(g_pc_.nodes))    
            # add an edge between the chosen node and the GCC node
            g_pc_.add_edge(random_node, gcc_node)               
            #caluculate the new components
            #curr_components = sorted(nx.connected_components(lattice_pc), key=len, reverse=True)
      
    components = sorted(nx.connected_components(g_pc_), key=len, reverse=True)
    curr_gcc_size = len(max(components))            
    deltaGcc += curr_gcc_size - gcc_size_at_pc  
    gccMass.append( deltaGcc )      
    #print(deltaGcc)
    # Update gcc
    gcc = nx.Graph(g_pc_.subgraph(components[0]))   
                 
    #%%



fig, ax = plt.subplots( )
# Enable usetex option
#plt.rcParams['text.usetex'] = True

ax.plot(p, GCC, label='GCC')
ax.plot(p, GCC2, label='GCC2')

# Add vertical line at p_c
ax.axvline(x=p_c, color='r', linestyle='--')

# Add text box with p_c value
ax.text(p_c, 0.1, f"p_c = {p_c:.4f}", ha='center', va='center',
        transform=ax.get_xaxis_transform())

# Add legend and axis labels
ax.legend()
ax.set_xlabel('p')
plt.show()

ax.set_ylabel('$P_{\infty}$')

#%% trail

num_values = 100
start = 0
end = 0.05

log_values = np.logspace(0, np.log10(end - start + 1), num_values)

# Generate logarithmically spaced values from 1 to (end - start + 1)

# Normalize the values to the desired range [start, end]
_= log_values - 1
p_values = np.diff(_)
print(p_values)
#np.savetxt("percolation_results.txt", np.column_stack((r, g)))
#%%

def generate_decreasing_density_list(n, base):
    """Generate a list of numbers with decreasing density between 0 and 1.

    Args:
        n (int): Number of elements in the list.
        base (float): Base of the exponential function, must be greater than 1.

    Returns:
        list: A list of n numbers with decreasing density between 0 and 1.
    """
    indices = np.arange(n)
    base = np.array(base).astype(float)
    indices = np.array(indices).astype(float)
    
    numbers = np.power(base, -indices)

    return numbers / (1 - 1 / base) 
n=np.logspace(0,0.05,50)-1
# Example usage
n = 50  # Number of elements in the list
base = 20  # Base of the exponential function

decreasing_density_list = generate_decreasing_density_list(n, base)
print(decreasing_density_list)

#%%
mask = (p >= p_c) & (p <= p_c + 0.05)
indices = list(np.where(mask)[0])
GCC_np = np.array(GCC)  # Convert GCC to a NumPy array
GCC_selected = GCC_np[indices]
p_selected = p[indices] - p_c
print(p_c)
#%%















#%%
#%%
#%%
#%% Fit - test
# Find indices where p_selected is not equal to zero
non_zero_indices = np.where(p_selected != 0)

# Remove zero values from p_selected and the corresponding entries in GCC_selected
p_selected_no_zero = p_selected[non_zero_indices]
GCC_selected_no_zero = GCC_selected[non_zero_indices]
from scipy.optimize import curve_fit

# Power-law function for fitting
def power_law(x, a, b):
    return a * x ** b

# Perform curve fitting
popt, _ = curve_fit(power_law, p_selected_no_zero, GCC_selected_no_zero)

# Calculate the fitted values
p_fit = np.logspace(np.log10(p_selected_no_zero.min()), np.log10(p_selected_no_zero.max()), 100)
GCC_fit = power_law(p_fit, *popt)

# Plot the log-log graph
plt.loglog(p_selected_no_zero, GCC_selected_no_zero, 'o', label='Data')
plt.loglog(p_fit, GCC_fit, '-', label=f'Fit: a={popt[0]:.3f}, b={popt[1]:.3f}')

plt.xlabel('$p-p_c$')
plt.ylabel('$P_{\infty}$')
plt.title('Log-log plot of $P_{\infty}$ vs. $p-p_c$')
plt.legend()
plt.show()

# The slope of the fitted line in the log-log scale is the value of the b parameter
slope = popt[1]
print(f"The slope of the fitted line is: {slope:.3f}")

#%%
L = 500
num_sims = 10

GCC_selected_list = []
p_selected_list = []
start_time = time.time()

for sim in range(num_sims):
    GCC, GCC2, p, p_c, g_pc = simulate_percolation_2d_lattice_return_gcc_gcc2_pc_gammaexp(L)
    mask = (p >= p_c) & (p <= p_c + 0.05)
    indices = np.where(mask)[0]
    GCC_selected = np.array(GCC, dtype=type(indices[0]))[indices]
    p_selected = p[indices] - p_c   
    GCC_selected_list.append(GCC_selected)
    p_selected_list.append(p_selected)


end_time = time.time()
running_time = end_time - start_time
print(f"Running time of simulation: {running_time:.3f} seconds")    
# %%

g_pc, p, p_c

nodes = list(g_pc)
      
# add p*N random edges
num_edges_to_add = int(p * N)
for i in range(num_edges_to_add):
    # randomly choose two nodes to connect
    node1, node2 = random.sample(nodes, 2)
    # add an edge between them
    G.add_edge(node1, node2)
  






