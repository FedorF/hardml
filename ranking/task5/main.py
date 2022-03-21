import numpy as np
import torch
from torch.nn import functional as F

from ranking.task5.knrm import KNRM, GaussianKernel, Solution
from ranking.task5.params import PATH_GLOVE, PATH_GLUE_QQP


def run_solution():
    solution = Solution(PATH_GLUE_QQP, PATH_GLOVE)
    emb_matrix, vocab, unk_words = solution.create_glove_emb_from_file(
        solution.glove_vectors_path, solution.all_tokens, solution.random_seed, solution.emb_rand_uni_bound)
    print(emb_matrix.shape)
    print(len(vocab))
    print(len(unk_words))
    print(len(unk_words) / len(emb_matrix))


def init():
    model = KNRM(np.array([[10,10]]), freeze_embeddings=False, out_layers=[7])
    print(model.mlp)


if __name__ == '__main__':
    init()
