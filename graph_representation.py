import networkx as nx
import numpy as np

def superpixels_to_graph(labels):
    num_labels = np.max(labels) + 1
    G = nx.Graph()
    
    for i in range(num_labels):
        G.add_node(i)
    
    # Add edges based on spatial adjacency (simple example)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            label = labels[i, j]
            if i > 0:
                neighbor_label = labels[i-1, j]
                if label != neighbor_label:
                    G.add_edge(label, neighbor_label)
            if j > 0:
                neighbor_label = labels[i, j-1]
                if label != neighbor_label:
                    G.add_edge(label, neighbor_label)
    
    return G
