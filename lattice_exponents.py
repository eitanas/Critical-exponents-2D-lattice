#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:18:57 2023

@author: Eitan
"""
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def create_random_lattice_with_pN_edges(L, p):
     
    N = L * L  # total number of nodes
    
    # create 2D lattice with N nodes
    G = nx.grid_2d_graph(L, L)
    G.remove_edges_from(list(G.edges()))
    nodes = list(G)
      
    # add p*N random edges
    num_edges_to_add = int(p * N)
    for i in range(num_edges_to_add):
        # randomly choose two nodes to connect
        node1, node2 = random.sample(nodes, 2)
        # add an edge between them
        G.add_edge(node1, node2)
      
    return G

L = 1000  # length of each side
N = L*L
p1 = 0.55 # fraction of edges to add
G = create_random_lattice_with_pN_edges(L,p1)

pos = {(i, j): (i, j) for i in range(L) for j in range(L)}   
labels = dict( ((i, j), i * 10 + j) for i, j in G.nodes() )

DRAW_NET=0
if DRAW_NET:    
    nx.draw_networkx(G, pos=pos, with_labels=False)
    plt.axis('off')
    plt.show()    
#%% Now, at each iteration we add p*N nodes to the graph
nodes = list(G)

step_p = 100/N
start_p = p1+step_p #Where I want to start with higher res of p
end_p = 0.1
num_ps = int(end_p/step_p)

p_values_close_to_criticallity = np.linspace(step_p, 0.1, num_ps )  # array of p values

# Initiate 
cluster_sizes_dict = {}
for p in p_values_close_to_criticallity:
    cluster_sizes_dict[p1+step_p]=[]

sim=1
# list to store graphs at each iteration
graph_list = []

for i in tqdm(range(num_ps)):
    if np.mod(i,100)==0: print(i)
    # add p*N random edges
    num_edges_to_add = int(step_p * N)
    for j in range(num_edges_to_add):
        # randomly choose two nodes to connect
        node1, node2 = random.sample(nodes, 2)
        # add an edge between them
        G.add_edge(node1, node2)
    # store the graph at each iteration
    graph_list.append(G.copy())    
    # calculate cluster size distribution
    connected_components = sorted(nx.connected_components(G), key=len)
    ccs = [len(gi)/N for gi in connected_components]
    cluster_sizes_dict[p1+i*step_p]= ccs

    
#%%        
# Get a list of keys in the dictionary
p_keys = list(cluster_sizes_dict.keys())
# Sort the keys in ascending order
p_keys.sort()

# Create lists to hold the largest and second largest clusters
largest_clusters = []
second_largest_clusters = []

# Iterate over the keys and append the largest and second largest clusters to the lists
for p in p_keys:
    # Get the cluster size distribution for the current key
    cluster_sizes = cluster_sizes_dict[p]
    # Calculate the sizes of the largest and second largest clusters
    largest = np.max(cluster_sizes)
    second_largest = sorted(cluster_sizes, reverse=True)[1]
    # Append the sizes to the lists
    largest_clusters.append(largest)
    second_largest_clusters.append(second_largest)

# Plot the largest and second largest clusters as a function of p
plt.plot(p_keys, largest_clusters, label='Largest')
plt.plot(p_keys, second_largest_clusters, label='Second largest')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('p')
plt.ylabel('Cluster size')
plt.legend()
plt.show()

#%% Now, from the second gcc we can find p_c

idx_phase_trans = np.argmax(second_largest_clusters)
p_c = p_keys[idx_phase_trans]
print(p_c)

#%% Delta exponent
# The algorithm is as follows:
# I define rho, which will be the fraction of nodes that 
# I test at each iteration.
# I choose randomly rho*N number of nodes
# for each node I chose, check whether or not it belongs 
# to the giant component
# If it doesnt belong, I connect it to one of the nodes in teh gcc.
# Then plot the Mass of the giant as a function of rho
# Subtract the gcc size from the Mass
# The gcc size corresponds to the one that was found by the size of the second gcc for each iteration

# I want to take the graphs starting at the critical point, and no larger than 0.05p 
# How many points I need to take until I cover 0.05p? 0.05/step_size
num_points_to_take_for_exp = int(0.05/step_p) # Number of points we wish to take in order to 

gs_for_exponent = graph_list[idx_phase_trans:idx_phase_trans+num_points_to_take_for_exp]

g_pc = gs_for_exponent[idx_phase_trans]
gcc_pc_graph = max(nx.connected_components(g_pc), key=len)
gcc_pc_subgraph = g_pc.subgraph(gcc_pc_graph)
gcc = gcc_pc_subgraph.copy()

#connected_components_pc = sorted(nx.connected_components(g_pc), key=len)
#ccs_pc = [len(gi)/N for gi in connected_components]

gcc_pc_size_0 = len(gcc)
gcc_size = gcc_pc_size_0

gcc_frac_pc_0=gcc_size/N
# gcc_pc_graph_0 = max(nx.connected_components(g_pc), key=len)

rho_step = 1/N
gccMass=[0]

for i in tqdm(range(num_points_to_take_for_exp)):
    
    num_nodes_to_test = int(rho_step*N)
    
    for j in range(num_nodes_to_test):
        
        # get a random node
        random_node = random.choice(list(g_pc.nodes()))

        # test if this node belongs to the gcc
        if random_node not in gcc:
            
            gcc_node = random.choice(list(gcc.nodes))

            # add an edge between the chosen node and the GCC node
            g_pc.add_edge(random_node, gcc_node)
            
            #caluculate
            curr_ccs = sorted(nx.connected_components(g_pc), key=len, reverse=True)
            
            curr_gcc_size = len(max(curr_ccs))
            gccMass.append(  curr_gcc_size - gcc_pc_size_0 )
            
            # Update the values of g_pc
            gcc = max(nx.connected_components(g_pc), key=len)
            gcc = g_pc.subgraph(gcc)
        else:
            gccMass.append(gccMass[-1])
# Plot the gcc mass as function of rho
#%%
plt.figure()
# Enable LaTeX rendering for labels
plt.rc('text', usetex=True)

rho = np.arange(0, rho_step * num_points_to_take_for_exp, step=rho_step)

# plot scatter plot
x, y = rho, gccMass
plt.scatter(x, y)

# add linear fit
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

# calculate slope (exponent delta)
delta = z[0]

# show plot and slope value
plt.xlabel('$\rho$')
plt.ylabel('Gcc mass')
plt.title('Change of Gcc mass as a function of rho')
plt.show()

print(f"Slope (delta) = {delta}")

#%%
x=2





