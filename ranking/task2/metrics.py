from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    _, indices = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[indices]
    cnt = 0
    for i, y in enumerate(ys_true_sorted[:-1]):
        for x in ys_true_sorted[i:]:
            cnt += int(x > y)

    return cnt


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2 ** y_value - 1
    pass


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, indices = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[indices]
    gain = 0
    for i, y in enumerate(ys_true_sorted, start=1):
        gain += compute_gain(y.item(), gain_scheme) / log2(i+1)

    return gain


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)

    return dcg(ys_true, ys_pred, gain_scheme) / ideal_dcg


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_pred.sum() == 0:
        return -1

    _, indices = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[indices]
    ind = min((len(ys_true), k))
    true_positives = ys_true_sorted[:ind].sum().item()
    positives = min((ys_true.sum().item(), k))

    return true_positives / positives


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, indices = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[indices]
    rank = 1 + (ys_true_sorted == 1).nonzero(as_tuple=True)[0].item()

    return 1 / rank


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    _, indices = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[indices]
    pfound, plook_prev, prel_prev = ys_true_sorted[0].item(), 1, ys_true_sorted[0].item()
    if len(ys_true_sorted) == 1:
        return pfound

    for y in ys_true_sorted[1:]:
        plook = plook_prev * (1 - prel_prev) * (1 - p_break)
        pfound += plook * y.item()
        plook_prev, prel_prev = plook, y.item()

    return pfound


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if ys_true.sum() == 0:
        return -1
    _, indices = sort(ys_pred, descending=True)
    ys_true_sorted = ys_true[indices]
    av_prec = 0
    for i, y in enumerate(ys_true_sorted, start=1):
        if y > 0:
            positives = ys_true_sorted[:i].sum().item()
            av_prec += positives / i
    av_prec /= ys_true.sum().item()

    return av_prec
