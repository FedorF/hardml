import numpy as np
from sklearn.metrics import euclidean_distances

from ranking.task6.nsw import distance, create_sw_graph, nsw


if __name__ == '__main__':
    a = np.random.randn(1, 10)
    b = np.random.randn(1000, 10)
    diff = distance(a, b) - euclidean_distances(a, b).reshape(len(b), -1)
    assert all(diff) < 1e-5
