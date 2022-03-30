from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    dist = np.sqrt(np.sum((pointA - documents) ** 2, axis=-1))
    return dist[..., np.newaxis]


def create_sw_graph(data: np.ndarray,
                    num_candidates_for_choice_long: int = 10,
                    num_edges_long: int = 5,
                    num_candidates_for_choice_short: int = 10,
                    num_edges_short: int = 5,
                    use_sampling: bool = False,
                    sampling_share: float = 0.05,
                    dist_f: Callable = distance,
                    ) -> Dict[int, List[int]]:
    data_size = len(data)
    sample_size = int(sampling_share * data_size)
    candidates = np.arange(data_size).flatten()

    graph = {}
    for i, point in tqdm(enumerate(data)):
        point = point[np.newaxis, ...]
        if use_sampling:
            candidates = np.random.choice(np.arange(data_size).flatten(), sample_size, replace=False)

        indices = dist_f(point, data[candidates]).flatten().argsort()
        indices = indices[indices != i]
        short_edges = indices[:num_candidates_for_choice_short]
        short_edges = np.random.choice(short_edges, num_edges_short, replace=False)

        long_edges = indices[-num_candidates_for_choice_long:]
        long_edges = np.random.choice(long_edges, num_edges_long, replace=False)

        graph[i] = [*short_edges, *long_edges]

    return graph


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10,
        num_start_points: int = 5,
        dist_f: Callable = distance,
        ) -> np.ndarray:
    stop_iter = 3  # todo: vary
    num_start_points = max(num_start_points, search_k)
    all_nodes = list(graph_edges.keys())
    start_nodes = np.random.choice(all_nodes, num_start_points, replace=False)
    knn = []
    for node in tqdm(start_nodes):
        # initialize random node and distance to it
        nn = np.random.randint(0, len(all_documents), 1)[0]
        min_dist = dist_f(query_point, all_documents[nn]).min()
        for i in range(stop_iter):
            edges = graph_edges[node]
            dists = dist_f(query_point, all_documents[edges])
            cur_min_dist, edge_min = dists.min(), edges[dists.flatten().argmin()]
            if cur_min_dist <= min_dist:
                min_dist = cur_min_dist
                nn = edge_min
            node = edge_min
        knn.append(nn)

    return np.array(knn)
