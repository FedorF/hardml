import numpy as np


def read_np_array(path):
    with open(path, 'rb') as file:
        array = np.load(file)

    return array
