import numpy as np
import torch
from torch.nn import functional as F

from ranking.task5.knrm import KNRM, GaussianKernel, Solution
from ranking.task5.params import PATH_GLOVE, PATH_GLUE_QQP


def run_solution():
    solution = Solution(PATH_GLUE_QQP, PATH_GLOVE)
    solution.train(12)


if __name__ == '__main__':
    run_solution()
