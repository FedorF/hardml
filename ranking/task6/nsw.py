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
    pass


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10,
        num_start_points: int = 5,
        dist_f: Callable = distance,
        ) -> np.ndarray:
    pass
