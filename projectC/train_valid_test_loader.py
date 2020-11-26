"""

Usage
-----
>>> train, valid, test, n_users, n_items = load_train_valid_test_datasets()
>>> n_users
943
>>> n_items
1682
>>> len(train[0]) # num train examples
70000
>>> len(valid[0]) # num valid examples
9992
>>> len(test[0])
10000
"""

import pandas as pd
import numpy as np
import os

def load_train_valid_test_datasets(
        data_path='data_movie_lens_100k/',
        n_valid=9992, # don't change this!
        n_test=10000, # don't change this!
        total_n_users=943,
        total_n_items=1682):
    ''' Load predefined train, valid, and test datasets from CSV file on disk.

    Returns
    -------
    train_data : tuple
    valid_data : tuple
    test_data : tuple
    total_n_users : int
    total_n_items : int
    '''

    ## Load the development set
    try:
        # Try first, in case directory is wrong, one too deep
        all_df = pd.read_csv(
            os.path.join('..', data_path, "ratings_all_development_set.csv"))
    except IOError:
        all_df = pd.read_csv(
            os.path.join(data_path, "ratings_all_development_set.csv"))
    L = all_df.shape[0]
    all_rows = np.arange(L)

    ## Divide into predefined splits
    test_rows = all_rows[:n_test]
    valid_rows = all_rows[n_test:(n_test+n_valid)]
    train_rows = all_rows[(n_test+n_valid):]

    assert np.intersect1d(train_rows, valid_rows).size == 0
    assert np.intersect1d(train_rows, test_rows).size == 0
    assert np.intersect1d(valid_rows, test_rows).size == 0

    all_data_tuple = (
        all_df['user_id'].values,
        all_df['item_id'].values,
        all_df['rating'].values)

    train_data_tuple = (
        all_df['user_id'].values[train_rows],
        all_df['item_id'].values[train_rows],
        all_df['rating'].values[train_rows])
    valid_data_tuple = (
        all_df['user_id'].values[valid_rows],
        all_df['item_id'].values[valid_rows],
        all_df['rating'].values[valid_rows])
    test_data_tuple = (
        all_df['user_id'].values[test_rows],
        all_df['item_id'].values[test_rows],
        all_df['rating'].values[test_rows])

    for dtuple in [all_data_tuple,
            train_data_tuple, valid_data_tuple, test_data_tuple]:
        assert np.all(dtuple[0] < total_n_users)
        assert np.all(dtuple[1] < total_n_items)

    return (
        train_data_tuple, valid_data_tuple, test_data_tuple,
        total_n_users, total_n_items)

