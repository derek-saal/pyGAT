import matplotlib.pyplot as plt
import networkx as nx
import gcn_utils

adj, _, labels, _, _, _ = gcn_utils.load_data('cora')

# Pre-processing for Networkx
nx_adj = nx.from_numpy_array(adj.ceil().numpy())
node_colour = labels.numpy() / max(labels)
labels = dict((node, label) for node, label in zip(range(len(labels)), labels.numpy()))

# Set options
options = {
    'node_size': 35,
    'font_size': 3,
    'width': 1.0,
    'alpha': 0.3,
    'labels': labels,
    'node_color': node_colour
}

# Draw graph
nx.draw_spectral(nx_adj, **options)
# nx.draw(nx_adj, **options)
plt.show()
