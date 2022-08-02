import numpy as np

from abt.task0 import confidence as conf


if __name__ == '__main__':
    p = 0.3
    a = (np.random.uniform(size=1000) > 1-p).astype(int)
    interval = conf.get_bernoulli_confidence_interval(a)
    print(interval)
