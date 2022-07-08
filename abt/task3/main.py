import numpy as np
import pandas as pd

from abt.task3 import mde

if __name__ == '__main__':
    result = mde.estimate_sample_size(
        df=pd.DataFrame(data={'revenue': 300 + 10*np.random.randn(1000)}),
        metric_name='revenue',
        effects=np.linspace(1.01, 1.1, 8),
    )
    print(result)
