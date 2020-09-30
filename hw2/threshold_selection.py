import numpy as np

from performance_metrics_for_binary_predictions import (
    calc_ACC, calc_TPR, calc_TNR, calc_PPV, calc_NPV)

def compute_perf_metrics_across_thresholds(ytrue_N, yproba1_N, thresh_grid=None):
    ''' Compute common binary classifier performance metrics across many thresholds
    
    If no array of thresholds is provided, will use all 'unique' values
    in the yproba1_N array to define all possible thresholds with different performance.
    
    Args
    ----
    ytrue_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    yproba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset

    Returns
    -------
    thresh_grid : 1D array of floats
        One entry for each possible threshold
    perf_dict : dict, with key, value pairs:
        * 'acc' : 1D array of accuracy values (one per threshold)
        * 'ppv' : 1D array of positive predictive values (one per threshold)
        * 'npv' : 1D array of negative predictive values (one per threshold)
        * 'tpr' : 1D array of true positive rates (one per threshold)
        * 'tnr' : 1D array of true negative rates (one per threshold)
    '''
    if thresh_grid is None:
        # Build default grid from 0.0 up to 1.000001 with 5 values
        default_thr_grid = np.linspace(0, 1.000001, 5)
        # Could threshold at each possible unique value of yproba
        unique_thr_grid = np.unique(yproba1_N)
        # Combine unique values with default ones to get reasonable coverage
        thresh_grid = np.sort(np.hstack([default_thr_grid, unique_thr_grid]))


    tpr_grid = np.zeros_like(thresh_grid)
    tnr_grid = np.zeros_like(thresh_grid)
    ppv_grid = np.zeros_like(thresh_grid)
    npv_grid = np.zeros_like(thresh_grid)
    acc_grid = np.zeros_like(thresh_grid)
    for tt, thresh in enumerate(thresh_grid):
        # Apply specific threshold to convert probas into hard binary values (0 or 1)
        yhat_N = np.asarray(yproba1_N >= thresh, dtype=np.int32)

        # Then count number of true positives, true negatives, etc.
        # Then compute metrics like accuracy and true positive rate
        acc_grid[tt] = calc_ACC(ytrue_N, yhat_N)
        tpr_grid[tt] = calc_TPR(ytrue_N, yhat_N)
        tnr_grid[tt] = calc_TNR(ytrue_N, yhat_N)
        ppv_grid[tt] = calc_PPV(ytrue_N, yhat_N)
        npv_grid[tt] = calc_NPV(ytrue_N, yhat_N)

    return thresh_grid, dict(
        acc=acc_grid,
        tpr=tpr_grid,
        tnr=tnr_grid,
        ppv=ppv_grid,
        npv=npv_grid)
