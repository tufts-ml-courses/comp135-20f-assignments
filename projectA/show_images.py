'''
Simple script to visualize 28x28 images stored in csv files

Usage
-----
$ python show_images.py --dataset_path data_sandal_vs_sneaker/

Expected Output
---------------
An active figure displaying 9 sample images arranged in 3x3 grid

'''

import argparse
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

def show_images(X, y, row_ids, n_rows=3, n_cols=3):
    ''' Display images

    Args
    ----
    X : 2D array, shape (N, 784)
        Each row is a flat image vector for one example
    y : 1D array, shape (N,)
        Each row is label for one example
    row_ids : list of int
        Which rows of the dataset you want to display
    '''
    fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            figsize=(n_cols * 3, n_rows * 3))

    for ii, row_id in enumerate(row_ids):
        cur_ax = axes.flatten()[ii]
        cur_ax.imshow(X[row_id].reshape(28,28), interpolation='nearest', vmin=0, vmax=1, cmap='gray')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        cur_ax.set_title('y=%d' % y[row_id])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='data_digits_8_vs_9_noisy')
    parser.add_argument('--n_rows', type=int, default=3)
    parser.add_argument('--n_cols', type=int, default=3)
    parser.add_argument('--example_ids_to_show', type=str, default='0,1,2,3,4,5,6,7,8')
    args = parser.parse_args()

    row_ids_to_show = list()
    for row_id_str in args.example_ids_to_show.split(','):
        row_id = int(row_id_str)
        row_ids_to_show.append(row_id)

    dataset_path = args.dataset_path

    x_df = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
    x_NF = x_df.values

    y_df = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
    y_N = y_df.values

    show_images(x_NF, y_N, row_ids_to_show, args.n_rows, args.n_cols)
    plt.show()

