def get_neighbors(src_node, edge_index, edge_weight = None):
    mask = edge_index[0] == src_node
    neighbors = edge_index[1][mask]
    neighbor_weights = edge_weight[mask] if edge_weight is not None else None
    return neighbors, neighbor_weights