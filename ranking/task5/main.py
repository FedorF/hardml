import numpy as np
import torch
from torch.nn import functional as F

from ranking.task5.knrm import KNRM, GaussianKernel, Solution
from ranking.task5.params import PATH_GLOVE, PATH_GLUE_QQP


def run_solution():
    solution = Solution(PATH_GLUE_QQP, PATH_GLOVE)
    solution.train(1)


def init():
    model = KNRM(np.array([[10,10]]), freeze_embeddings=False, out_layers=[7])
    print(model.mlp)


if __name__ == '__main__':
    run_solution()
