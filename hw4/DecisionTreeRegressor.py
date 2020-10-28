import numpy as np

from train_tree import train_tree_greedy

class MyDecisionTreeRegressor(object):

    '''
    Prediction model object implementing an sklearn-like interface.

    Allows fitting and predicting from a decision tree.

    Attributes
    ----------
    max_depth
    min_samples_leaf
    min_samples_split

    Attributes available after calling fit
    --------------------------------------
    root_node : Instance of InternalDecisionNode or LeafNode from tree_utils
        Supports predict method.

    Examples
    --------
    >>> N = 12

    >>> x_tr_N = np.asarray([
    ...     -0.975, -0.825, -0.603, -0.378, -0.284, -0.102,
    ...     0.169,  0.311,  0.431,  0.663,  0.795,  0.976])
    >>> x_tr_N1 = x_tr_N.reshape((N,1)) # need an (N,1) shaped array for later use with sklearn
    >>> y_tr_N = np.asarray([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

    >>> tree_predictor = MyDecisionTreeRegressor(
    ...     max_depth=2, min_samples_leaf=1, min_samples_split=2)
    >>> tree_predictor.fit(x_tr_N1, y_tr_N)

    # Display the trained tree
    >>> print(tree_predictor)
    Decision: X[0] < 0.034?
      Y: Decision: X[0] < -0.331?
          Y: Leaf: predict y = 0.500
          N: Leaf: predict y = 0.000
      N: Decision: X[0] < 0.547?
          Y: Leaf: predict y = 1.000
          N: Leaf: predict y = 0.333

    # Compare to sklearn, we should get the same tree
    >>> import sklearn.tree
    >>> from pretty_print_sklearn_tree import pretty_print_sklearn_tree
    >>> sktree = sklearn.tree.DecisionTreeRegressor(
    ...     max_depth=2, min_samples_leaf=1, min_samples_split=2)
    >>> sktree = sktree.fit(x_tr_N1, y_tr_N)
    >>> pretty_print_sklearn_tree(sktree, ['0'])
    The binary tree structure has 7 nodes.
    - depth   0 has    1 nodes, of which    0 are leaves
    - depth   1 has    2 nodes, of which    0 are leaves
    - depth   2 has    4 nodes, of which    4 are leaves
    The decision tree:  (Note: Y = 'yes' to above question; N = 'no')
    Decision: X['0'] <= 0.03?
      Y Decision: X['0'] <= -0.33?
        Y Leaf: yhat at this leaf = 0.500
        N Leaf: yhat at this leaf = 0.000
      N Decision: X['0'] <= 0.55?
        Y Leaf: yhat at this leaf = 1.000
        N Leaf: yhat at this leaf = 0.333
    <BLANKLINE>
    '''

    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=1):
        ''' Constructor for our prediction object

        Returns
        -------
        new MyDecisionTreeRegressor object
        '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x_NF, y_N):
        ''' Make prediction for each example in provided features array

        Args
        ----
        x_NF : 2D numpy array, shape (n_samples, n_features) = (N, F)
            Features for each training sample.
        y_N : 1D numpy array, shape (n_samples,) = (N,)
            Target outcome values for each training sample.

        Returns
        -------
        None.

        Post Condition
        --------------
        Attribute 'root_node' is set to a valid value.
        '''
        self.root_node = train_tree_greedy(
            x_NF, y_N, depth=0,
            MAX_DEPTH=self.max_depth,
            MIN_SAMPLES_INTERNAL=self.min_samples_split,
            MIN_SAMPLES_LEAF=self.min_samples_leaf)

    def predict(self, x_TF):
        ''' Make prediction for each example in provided features array.

        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
        '''
        return self.root_node.predict(x_TF)

    def to_string(self):
        return str(self.root_node)

    def __str__(self):
        return self.to_string()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    N = 12

    x_tr_N = np.asarray([
        -0.975, -0.825, -0.603, -0.378, -0.284, -0.102,
         0.169,  0.311,  0.431,  0.663,  0.795,  0.976])
    x_tr_N1 = x_tr_N.reshape((N,1)) # need an (N,1) shaped array for later use with sklearn
    y_tr_N = np.asarray([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

    L = 101
    x_dense_L1 = np.linspace(-1.5, 1.5, L).reshape((L,1))

    tree_predictor = MyDecisionTreeRegressor(
        max_depth=5, min_samples_leaf=1, min_samples_split=2)
    tree_predictor.fit(x_tr_N1, y_tr_N)

    print("Printing the best tree found by MyDecisionTreeRegressor:")
    print(tree_predictor)

    plt.plot(x_tr_N, y_tr_N, 'ks', label='Training set');
    plt.plot(x_tr_N, tree_predictor.predict(x_tr_N1), 'bd',
        label='Predictions on training set');
    plt.plot(x_dense_L1, tree_predictor.predict(x_dense_L1), 'b-', 
        label='Predictions evaluated at dense grid')

    plt.xlabel('x');
    plt.ylabel('y');
    plt.legend(bbox_to_anchor=(1.0, 0.5));
    plt.title("Predictions from MyDecisionTreeRegressor")
    plt.show()