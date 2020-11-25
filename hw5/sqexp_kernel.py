'''
Defines function `calc_sqexp_kernel`

Examples
--------
>>> np.set_printoptions(precision=3, suppress=1)

# Example 1: Simple kernel evaluations with F=1 features
>>> x_zero_11 = np.asarray([[0.0]])
>>> x_one_11 = np.asarray([[1.0]])
>>> k_11 = calc_sqexp_kernel(x_zero_11, x_zero_11, length_scale=1.0)
>>> k_11.ndim
2
>>> k_11
array([[1.]])
>>> calc_sqexp_kernel(x_one_11, x_one_11, length_scale=1.0)
array([[1.]])
>>> calc_sqexp_kernel(x_one_11, x_zero_11, length_scale=1.0)
array([[0.368]])


# Example 2: Kernel evaluations with F=2 features and several examples at once
>>> x_train_32 = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
>>> k_33 = calc_sqexp_kernel(x_train_32, x_train_32)
>>> k_33
array([[1.   , 0.135, 0.   ],
       [0.135, 1.   , 0.135],
       [0.   , 0.135, 1.   ]])

>>> x_test_22 = np.asarray([[-1.0, -2.0], [2.2, 0.8]])
>>> k_23 = calc_sqexp_kernel(x_test_22, x_train_32)
>>> k_23
array([[0.007, 0.   , 0.   ],
       [0.004, 0.228, 0.228]])
>>> 
'''

import numpy as np

def calc_sqexp_kernel(x_QF, x_train_NF=None, length_scale=1.0):
    ''' Evaluate squared-exponential kernel matrix between two datasets.

    Will compute the kernel function for all possible pairs of feature vectors,
    one from the query dataset, one from the reference training dataset.

	Args
	----
    x_QF : 2D numpy array, shape (Q, F) = (n_query_examples, n_features)
		Feature array for *query* dataset
		Each row corresponds to the feature vector on example

    x_train_NF : 2D numpy array, shape (N, F) = (n_train_examples, n_features)
		Feature array for reference *training* dataset
		Each row corresponds to the feature vector on example
		
    Returns
    -------
    k_QN : 2D numpy array, shape (Q, N)
        Entry at index (q,n) corresponds to the kernel function evaluated
        at the feature vectors x_QF[q] and x_train_NF[n]
    '''
    assert x_QF.ndim == 2
    assert x_train_NF.ndim == 2

    Q, F = x_QF.shape
    N, F2 = x_train_NF.shape
    assert F == F2

    k_QN = np.zeros((Q, N))
    # TODO compute kernel between rows of x_QF and rows of x_train_NF

    # Ensure the kernel matrix positive definite
    # By adding a small positive to the diagonal
    M = np.minimum(Q, N)
    k_QN[:M, :M] += 1e-08 * np.eye(M)
    return k_QN