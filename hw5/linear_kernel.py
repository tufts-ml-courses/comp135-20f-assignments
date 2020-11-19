'''
Defines function `calc_linear_kernel`

Examples
--------
>>> np.set_printoptions(precision=3, suppress=1)

# Kernel evaluations with F=1 features
>>> x_zero_11 = np.asarray([[0.0]])
>>> x_one_11 = np.asarray([[1.0]])

# Linear kernel k(0,0) should be zero
>>> k_11 = calc_linear_kernel(x_zero_11, x_zero_11)
>>> k_11.ndim
2
>>> k_11
array([[0.]])

# Linear kernel k(1,1) should be one
>>> calc_linear_kernel(x_one_11, x_one_11)
array([[1.]])

# Linear kernel k(1, 3.456) should be 3.456
>>> calc_linear_kernel(x_one_11, 3.456 * x_one_11)
array([[3.456]])


# Part 2: Kernel evaluations with F=2 features and several examples at once
>>> x_train_32 = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
>>> calc_linear_kernel(x_train_32, x_train_32)
array([[0., 0., 0.],
       [0., 2., 4.],
       [0., 4., 8.]])

>>> x_test_22 = np.asarray([[-1.0, -2.0], [2.2, 1.2]])
>>> calc_linear_kernel(x_test_22, x_train_32)
array([[ 0. , -3. , -6. ],
       [ 0. ,  3.4,  6.8]])

'''

import numpy as np

def calc_linear_kernel(x_QF, x_train_NF=None):
    ''' Evaluate linear kernel matrix between two datasets.

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
    # TODO compute linear kernel between rows of x_QF and rows of x_train_NF

    # Ensure the kernel matrix positive definite
    # By adding a small positive to the diagonal
    M = np.minimum(Q, N)
    k_QN[:M, :M] += 1e-08 * np.eye(M)
    return k_QN