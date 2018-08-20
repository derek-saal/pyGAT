import numpy as np


def modularity(adj, labels):
    n, _ = adj.shape
    deg = adj.ceil().sum(dim=1)
    m = deg.sum()
    q = 0.0
    for i in range(n):
        j = np.nonzero(adj[i])
        for j in range(n):
            q += (adj[i][j] - deg[i]*deg[j] / (2.0*m)) * _kd(labels[i], labels[j]) / (2.0 * m)
    return q


def _kd(i, j):
    """
    Kronecker delta
    """
    return int(i == j)


def participation_index(adj, labels):
    n, _ = adj.shape
    cs = np.unique(labels)  # communities
    deg = adj.ceil().sum(dim=1)
    pis = np.zeros(n)
    for i in range(n):
        c_counter = np.zeros(cs.size)
        if i % 100 == 0:
            print("Calculating participation index for node {}".format(i))
        for j in range(n):
            if adj[i][j]:
                c_counter[labels[j]] += 1.0
        for c in cs:
            pis[i] -= np.power(c_counter[c] / deg[i], 2)
        pis[i] += 1.0
    return pis


def hub_nodes(adj, labels):
    deg = adj.ceil().sum(dim=1)
    std_deg = deg.std()
    mean_deg = deg.mean()
    hubs = np.zeros(deg.shape[0])
    pi = participation_index(adj, labels)
    for i, d in enumerate(deg):
        if d > mean_deg + std_deg:
            hubs[i] = pi[i]
    return hubs
