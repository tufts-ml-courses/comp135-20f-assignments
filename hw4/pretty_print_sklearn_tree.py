import numpy as np
from collections import defaultdict

def pretty_print_sklearn_tree(tree_clf, feature_names):
    ''' Print out a nice summary of the provided tree.

    Args
    ----
    tree_clf : instance of sklearn.tree.DecisionTreeClassifier
    feature_names : list of strings
        Provides a printable 'name' for each feature the model was trained with.

    Returns
    -------
    Nothing. Results printed to stdout.
    
    Notes
    -----
    Based on an sklearn tutorial. Here's a brief summary of how it works:

    # After training, any DecisionTreeClassifier an attribute called tree_ 
    # which stores the tree structure and allows access to key attributes.
    # The binary tree tree_ is represented as a number of parallel arrays.
    # The i-th element of each array holds information about the node `i`.
    # Node 0 is the tree's root.
    # Notes:
    # - Some of the arrays only apply to either leaves or split nodes, resp.
    # - In this case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #   - value, counts of each class for train examples reaching this node

    By manipulating these arrays, we can parse the tree structure.
    '''

    n_nodes = tree_clf.tree_.node_count
    children_left = tree_clf.tree_.children_left
    children_right = tree_clf.tree_.children_right
    feature = tree_clf.tree_.feature
    threshold = tree_clf.tree_.threshold

    # The tree structure can be traversed to compute various properties
    # such as:
    # * the depth of each node 
    # * whether or not it is a leaf.

    node_depth_N = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaf_N = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth_N[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            # Internal decision node
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            # Leaf node
            is_leaf_N[node_id] = True

    print("The binary tree structure has %s nodes." % n_nodes)
    depths_U, counts_U = np.unique(node_depth_N, return_counts=True)
    for uu in range(depths_U.size):
        is_at_cur_depth_N = depths_U[uu] == node_depth_N
        is_leaf_at_cur_depth_N = np.logical_and(
            is_leaf_N,
            is_at_cur_depth_N,
            )
        print("- depth %3d has %4d nodes, of which %4d are leaves" % (
            depths_U[uu], counts_U[uu], np.sum(is_leaf_at_cur_depth_N)))

    print("The decision tree:  (Note: Y = 'yes' to above question; N = 'no')")
    n_seen_by_depth = defaultdict(int)

    for i in range(n_nodes):
        cur_depth = node_depth_N[i]
        count_at_cur_depth = n_seen_by_depth[cur_depth]

        # Every other printout at same depth should alternate Y and N labels
        if node_depth_N[i] == 0:
            decision_str = '' # base case
        elif count_at_cur_depth % 2 == 0:
            decision_str = 'Y '
        else:
            decision_str = 'N '

        if is_leaf_N[i]:
            n_class0 = tree_clf.tree_.value[i,0,0]
            n_class1 = tree_clf.tree_.value[i,0,1]
            proba1 = n_class1 / (n_class1 + n_class0)
            print("%s%sLeaf: p(y=1 | this leaf) = %.3f (%d total training examples)" % (
                node_depth_N[i] * "  ", decision_str, proba1, n_class0 + n_class1))

        else:
            print("%s%sDecision: X['%s'] <= %.2f?" % (
                node_depth_N[i] * "  ",
                decision_str,
                feature_names[feature[i]],
                threshold[i],
                ))

        # Increment our counter so we get the alternating Y/N labels right
        n_seen_by_depth[cur_depth] += 1

    print()

