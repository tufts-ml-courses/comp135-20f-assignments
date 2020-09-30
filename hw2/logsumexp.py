'''
logsumexp.py

Provides a numerically implementation of logsumexp function,
such that no matter what 1-dimensional input array is provided,
we return a finite floating point answer that does not overflow or underflow.

References
----------
See the math here:
https://www.cs.tufts.edu/comp/135/2020f/hw2.html#logsumexp
'''

import numpy as np

# No other imports allowed

def my_logsumexp(scores_N):
    ''' Compute logsumexp on provided array in numerically stable way.

    This function only handles 1D arrays.
    The equivalent scipy function can handle arrays of many dimensions.

    Args
    ----
    scores_N : 1D NumPy array, shape (N,)
        An array of real values

    Returns
    -------
    a : float
        Result of the logsumexp computation

    Examples
    --------
    >>> _ = np.seterr(all='raise') # Make any numerical issue raise error

    # Example 1: an array without overflow trouble, so you get the basic idea
    >>> easy_arr_N = np.asarray([0., 1., 0.1])

    # Show that your code does OK on Example 1
    >>> easy_ans = my_logsumexp(easy_arr_N)
    >>> print("%.5f" % (easy_ans))
    1.57349

    # Show that naive implementation does OK on Example 1
    >>> naive_ans = np.log(np.sum(np.exp(easy_arr_N)))
    >>> print("%.5f" % (naive_ans))
    1.57349

    # Example 2: an array where overflow would occur in bad implementation
    >>> tough_arr_N = [1000., 1001., 1002.]

    # Show that naive implementation suffers from overflow on Example 2
    >>> naive_ans = np.log(np.sum(np.exp(tough_arr_N)))
    Traceback (most recent call last):
    ...
    FloatingPointError: overflow encountered in exp

    # Show that your implementation does well on Example 2
    >>> ans_that_wont_overflow = my_logsumexp(tough_arr_N)
    >>> np.isfinite(ans_that_wont_overflow)
    True
    >>> print("%.5f" % (ans_that_wont_overflow))
    1002.40761
    '''
    scores_N = np.asarray(scores_N, dtype=np.float64)

    # TODO compute logsumexp in numerically stable way
    # See math on HW2 instructions page for the correct approach
    return 0.0