import numpy as np

from tree_utils import LeafNode, InternalDecisionNode
from select_best_binary_split import select_best_binary_split

def train_tree_greedy(
        x_NF, y_N, depth,
        MAX_DEPTH=10,
        MIN_SAMPLES_INTERNAL=1,
        MIN_SAMPLES_LEAF=1):
    ''' Train a binary decision tree on provided dataset in greedy fashion.

    Args
    ----
    x_NF : 2D array, shape (N,F) = (n_examples, n_features)
        Training data features to use to build the tree.
    y_N : 1D array, shape (N,) = (n_examples,)
        Training labels to use to build the tree.
    depth : int
        Current depth of the tree. Needed to use as a recursive function.
        Should be zero when called by end user.
    MAX_DEPTH : int
        Hyperparameter controlling maximum allowed depth.
        Smaller values might protect against overfitting.
    MIN_SAMPLES_INTERNAL : int
        Controls minimum samples allowed for an InternalDecisionNode.
        Higher values might protect against overfitting.
    MIN_SAMPLES_LEAF : int
        Hyperparameter controlling minimum samples allowed for a leaf node.
        Higher values might protect against overfitting.

    Returns
    -------
    root_node : an instance of LeafNode or InternalDecisionNode
        This node determines the root of the requested tree.
        You can query its attributes `left_child` and `right_child` to get
        other nodes in the tree (if they exist).

    '''
    N, F = x_NF.shape
    MIN_SAMPLES_INTERNAL = np.maximum(
        MIN_SAMPLES_INTERNAL, 2 * MIN_SAMPLES_LEAF)
    if depth >= MAX_DEPTH:
        return LeafNode(x_NF, y_N)
    elif N < MIN_SAMPLES_INTERNAL:
        return LeafNode(x_NF, y_N)
    else:
        feat_id, thresh_val, x_LF, y_L, x_RF, y_R = select_best_binary_split(
                x_NF, y_N, MIN_SAMPLES_LEAF)
        if feat_id is None:
            # Case where further split is not possible
            return LeafNode(x_NF, y_N)
        else:
            # TODO recursively call train_tree_greedy to build the left child
            left_child = None # TODO fixme
            # TODO recursively call train_tree_greedy to build the right child
            right_child = None # TODO fixme
            return InternalDecisionNode(
                x_NF, y_N, feat_id, thresh_val, left_child, right_child)