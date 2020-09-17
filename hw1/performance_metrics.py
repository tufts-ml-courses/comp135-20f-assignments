import numpy as np


def calc_mean_squared_error(y_N, yhat_N):
    ''' Compute the mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mse : scalar float
        Mean squared error performance metric
        .. math:
            mse(y, \hat{y}) = \frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2

    Examples
    --------
    >>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
    >>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
    >>> calc_mean_squared_error(y_N, yhat_N)
    1.3333333333333333
    '''
    return 0.0  # TODO fixme


def calc_mean_absolute_error(y_N, yhat_N):
    ''' Compute the mean absolute error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mae : scalar float
        Mean absolute error performance metric
        .. math:
            mae(y, \hat{y}) = \frac{1}{N} \sum_{n=1}^N | y_n - \hat{y}_n |

    Examples
    --------
    >>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
    >>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
    >>> calc_mean_absolute_error(y_N, yhat_N)
    0.6666666666666666
    '''
    return 0.0  # TODO fixme
