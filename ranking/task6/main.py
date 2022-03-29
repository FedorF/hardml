import numpy as np
from sklearn.metrics import euclidean_distances

from ranking.task6.nsw import create_sw_graph, nsw


if __name__ == '__main__':
    queries, documents = np.random.randn(100, 2), np.random.randn(2000, 2)
    graph = create_sw_graph(documents, use_sampling=True)
    knn = nsw(queries, documents, graph)
