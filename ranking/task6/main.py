import numpy as np

from ranking.task6.nsw import create_sw_graph, nsw


if __name__ == '__main__':
    query, documents = np.random.randn(1, 5), np.random.randn(1000, 5)
    graph = create_sw_graph(documents)
    knn = nsw(query, documents, graph)
    print(knn)
