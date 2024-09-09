import torch
import torch_geometric as pyg

def wavepool(graph):
    data = pyg.data.Data(edge_index=graph.edges, num_nodes=len(graph.nodes))
    pooled_data = pyg.nn.GlobalAttention()(data)
    return pooled_data
