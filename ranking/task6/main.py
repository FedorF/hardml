import numpy as np
from sklearn.metrics import euclidean_distances

from ranking.task6.nsw import create_sw_graph, nsw


if __name__ == '__main__':
    query, documents = np.random.randn(1, 5), np.random.randn(10000, 5)
    graph = create_sw_graph(documents, use_sampling=True)
    knn = nsw(query, documents, graph)
