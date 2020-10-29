import numpy as np

from tree_utils import LeafNode, InternalDecisionNode

def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset

    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features at current node we wish to find a split for.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels at current node.
    min_samples_leaf : int
        Minimum number of samples allowed at any leaf.

    Returns
    -------
    feat_id : int or None, one of {0, 1, 2, .... F-1}
        Indicates which feature in provided x array is used for best split.
        If None, a binary split that improves the cost is not possible.
    thresh_val : float or None
        Value of x[feat_id] at which we threshold.
        If None, a binary split that improves the cost is not possible.
    x_LF : 2D array, shape (L, F)
        Training data features assigned to left child using best split.
    y_L : 1D array, shape (L,)
        Training labels assigned to left child using best split.
    x_RF : 2D array, shape (R, F)
        Training data features assigned to right child using best split.
    y_R : 1D array, shape (R,)
        Training labels assigned to right child using best split.

    Examples
    --------
    # Example 1a: Simple example with F=1 and sorted features input
    >>> N = 6
    >>> F = 1
    >>> x_NF = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 1b: Same as 1a but just scramble the order of x
    # Should give same results as 1a
    >>> x_NF = np.asarray([2.0, 1.0, 0.0, 3.0, 5.0, 4.0]).reshape((6, 1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    0
    >>> thresh_val
    2.5

    # Example 2: Advanced example with F=12 total features
    # Fill the features such that middle column is same as 1a above,
    # but the first 6 columns with random features
    # and the last 6 columns with all zeros
    >>> N = 6
    >>> F = 13
    >>> prng = np.random.RandomState(0)
    >>> x_N1 = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape((6,1))
    >>> x_NF = np.hstack([prng.randn(N, F//2), x_N1, np.zeros((N, F//2))])
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id
    6
    >>> thresh_val
    2.5

    # Example 3: binary split isn't possible (because all x same)
    >>> N = 5
    >>> F = 1
    >>> x_NF = np.asarray([3.0, 3.0, 3.0, 3.0, 3.0]).reshape((5,1))
    >>> y_N  = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0])
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True

    # Example 4: binary split isn't possible (because all y same)
    >>> N = 5
    >>> F = 3
    >>> prng = np.random.RandomState(0)
    >>> x_NF = prng.rand(N, F)
    >>> y_N  = 1.2345 * np.ones(N)
    >>> feat_id, thresh_val, _, _, _, _ = select_best_binary_split(x_NF, y_N)
    >>> feat_id is None
    True
    '''
    N, F = x_NF.shape

    # Allocate space to store the cost and threshold of each feat's best split
    cost_F = np.inf * np.ones(F)
    thresh_val_F = np.zeros(F)
    for f in range(F):

        # Compute all possible x threshold values for current feature
        # possib_xthresh_V : 1D array of size V
        #    Each entry is a float
        #    Entries are in sorted order from smallest to largest
        #    Represents one possible distinct threshold for provided N examples
        xunique_U = np.unique(x_NF[:,f])
        possib_xthresh_V = 0.5 * (
            xunique_U[MIN_SAMPLES_LEAF:] + xunique_U[:-MIN_SAMPLES_LEAF])
        V = possib_xthresh_V.size
        if V == 0:
            # If all the x values for this feature are same, we can't split.
            # Keep cost as "infinite" and continue to next feature
            cost_F[f] = np.inf
            continue

        # TODO Compute total cost at each possible threshold
        # Hint: You may need several lines of code, and maybe a for loop.
        # Your goal is *correctness*, don't prioritize speed or efficiency yet.
        left_yhat_V = np.zeros(V) # TODO fixme
        right_yhat_V = np.ones(V) # TODO fixme
        left_cost_V = np.zeros(V) # TODO fixme
        right_cost_V = np.ones(V) # TODO fixme
        total_cost_V = left_cost_V + right_cost_V

        # Check if there is any split that improves our cost or predictions.
        # If not, all splits will have same cost and we should just not split.
        costs_all_the_same = np.allclose(total_cost_V, total_cost_V[0])
        yhat_all_the_same = np.allclose(left_yhat_V, right_yhat_V)
        if costs_all_the_same and yhat_all_the_same:
            # Keep cost as "infinite" and continue to next feature
            cost_F[f] = np.inf
            continue
        
        # TODO pick out the split candidate that has best cost
        chosen_v_id = -1 # TODO fixme
        cost_F[f] = total_cost_V[chosen_v_id]
        thresh_val_F[f] = possib_xthresh_V[chosen_v_id]

    # Determine single best feature to use
    best_feat_id = np.argmin(cost_F)
    best_thresh_val = thresh_val_F[best_feat_id]
    
    if not np.isfinite(cost_F[best_feat_id]):
        # Edge case: not possible to split further
        # Either all x values are the same, or all y values are the same
        return (None, None, None, None, None, None)

    ## TODO assemble the left and right child datasets
    x_LF, y_L = None, None # TODO fixme
    x_RF, y_R = None, None # TODO fixme

    # TODO should verify your cost computation:
    # Just to make sure you've done everything right.
    # >>> left_cost = np.sum(np.square(y_L - np.mean(y_L)))
    # >>> right_cost = np.sum(np.square(y_R - np.mean(y_R)))
    # >>> assert np.allclose(cost_F[best_feat_id], left_cost + right_cost)

    return (best_feat_id, best_thresh_val, x_LF, y_L, x_RF, y_R)