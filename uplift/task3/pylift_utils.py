import numpy as np
import pandas as pd
import pylift
import matplotlib.pyplot as plt

from typing import List, Tuple


METRICS = [
    'q2_cgains',
    'q2_qini',
    'q2_aqini',
    'q1_cgains',
    'q1_qini',
    'q1_aqini',
    'Q_cgains',
    'Q_qini',
    'Q_aqini',
    'Q_max',
    'Q_practical_max',
]


def get_metrics_table(
    upeval: pylift.eval.UpliftEval, metrics: List[str] = METRICS
) -> pd.DataFrame:
    values = [upeval.__getattribute__(metric) for metric in metrics]
    return pd.DataFrame(data={'metric': metrics, 'value': values})


def make_report_(
    upeval: pylift.eval.UpliftEval,
    plots: List[str] = ['balance', 'cgains', 'uplift'],
    metrics: List[str] = METRICS
) -> None:
    display(get_metrics_table(upeval, metrics))

    for plot in plots:
        upeval.plot(plot)


def make_report(
    data_test: pd.DataFrame,
    treatment: List[float],
    outcome: List[float],
    model = None,
    prediction: List[float] = None,
    plots: List[str] = ['balance', 'cgains', 'uplift'],
    metrics: List[str] = METRICS,
    uplift_eval_args = {}
) -> Tuple[pylift.eval.UpliftEval, List[float]]:
    
    if prediction is None:
        prediction = model.predict(data_test)
        
    upeval = pylift.eval.UpliftEval(
        treatment=treatment,
        outcome=outcome,
        prediction=prediction,
        **uplift_eval_args
    )
    
    make_report_(upeval, plots, metrics)
    
    return upeval, prediction


def plot_uplift_prediction(
    upeval: pylift.eval.UpliftEval, plot_type: str = 'uplift',
    n_bins: int = 20, do_plot: bool = True
):
    bin_range = np.linspace(0, len(upeval.treatment), n_bins+1).astype(int)
    
    def noncumulative_subset_func(i):
        return np.isin(list(range(len(upeval.treatment))), prob_index[bin_range[i]:bin_range[i+1]])
    def cumulative_subset_func(i):
        return np.isin(list(range(len(upeval.treatment))), prob_index[:bin_range[i+1]])

    subsetting_functions = {
        'cuplift': cumulative_subset_func,
        'uplift': noncumulative_subset_func,
    }

    prob_index = np.flip(np.argsort(upeval.prediction), 0)
    
    x = list()
    y = list()
    
    for i in range(n_bins):
        current_subset = subsetting_functions[plot_type](i)
        
        # Get the values of outcome in this subset for test and control.
        treated_subset = (upeval.treatment == 1) & current_subset
        untreated_subset = (upeval.treatment == 0) & current_subset
        
        # Get the policy for each of these as well.
        p_treated = upeval.p[treated_subset]
        p_untreated = upeval.p[untreated_subset]

        # Count the number of correct values (i.e. y==1) within each of these
        # sections as a fraction of total ads shown.
        nt1 = np.sum(0.5 / p_treated)
        nt0 = np.sum(0.5 / (1 - p_untreated))
        
        y.append(upeval.prediction[current_subset].mean())
        x.append(nt1 + nt0)
    
    
    x = np.cumsum(x)
    # Rescale x so it's between 0 and 1.
    percentile = x / np.amax(x)

    # percentile = np.insert(percentile, 0, 0)
    # y.insert(0,0)
    
    if do_plot:
        plt.plot(percentile, y)
    
    return percentile, y
