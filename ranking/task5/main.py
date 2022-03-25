import numpy as np
import torch
from torch.nn import functional as F

from ranking.task5.knrm import KNRM, GaussianKernel, Solution
from ranking.task5.params import PATH_GLOVE, PATH_GLUE_QQP
import string

def run_solution():
    solution = Solution(PATH_GLUE_QQP, PATH_GLOVE)
    solution.train(1)

def _kernel_sigmas(n_kernels, sigma, exact_sigma, lamb=None):
    l_sigma = [exact_sigma]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    # use different sigmas for kernels
    if lamb:
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma += [bin_size * lamb] * (n_kernels - 1)
    else:
        for i in range(1, n_kernels):
            l_sigma.append(sigma)

    return l_sigma


if __name__ == '__main__':
    # run_solution()
    print(list(map(int, _kernel_sigmas(21, sigma= 0.1, exact_sigma= 0.001,))))