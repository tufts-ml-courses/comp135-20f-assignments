import numpy as np
import pandas as pd

import sklearn.metrics

import matplotlib.pyplot as plt
import matplotlib.colors

def plot_pretty_probabilities_for_clf(
        clf, x_tr_N2, y_tr_N,
        ax=None,
        x1_grid=(-2, 2, 50), x2_grid=(-2, 2, 50),
        x1_ticks=[-1, 0, 1], x2_ticks=[-1, 0, 1],
        do_show_colorbar=False,
        c_ticks=np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        c_num_levels=100,
        ):
    ''' Display predicted probabilities from classifier as color contour plot

    Args
    ----
    clf : sklearn object that implements classifier API
    x_tr_N2 : 2D array, size N x 2
        Features for each training example (a length-2 vector)
    y_tr_N : 1D array, size N
        Labels for each training example (binary, 0 or 1)
    ax : matplotlib axis handle, or None
        If provided, will use axis handle as primary axis to plot on.
        If None, will use the current handle, or make new one as necessary.
    x1_grid : tuple-like or array-like
        If tuple of length 3, interpreted as args to np.linspace
        Otherwise, cast to array and assumed to be a 1d grid of x1 values
    x2_grid : tuple-like or array-like
        If tuple of length 3, interpreted as args to np.linspace
        Otherwise, cast to array and assumed to be a 1d grid of x2 values

    '''
    # Activate the current axis, if necessary
    if ax is None:
        cur_ax = plt.gca()
    else:
        cur_ax = ax
        plt.sca(cur_ax)

    # Plot the training data, colored by true label
    cur_ax.plot(x_tr_N2[y_tr_N==0, 0], x_tr_N2[y_tr_N==0,1], 'rx', mew=2)
    cur_ax.plot(x_tr_N2[y_tr_N==1, 0], x_tr_N2[y_tr_N==1,1], 'b+', mew=2)

    # Make predictions and compute error rates and losses
    tr_loss = sklearn.metrics.log_loss(y_tr_N, clf.predict_proba(x_tr_N2)[:,1])
    tr_err = sklearn.metrics.zero_one_loss(y_tr_N, clf.predict(x_tr_N2))
    cur_ax.set_title(
        'log loss %.3f  err_rate %.3f' % (tr_loss, tr_err))

    # Visualize predicted probabilities
    if isinstance(x1_grid, tuple) and len(x1_grid) == 3:
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], x1_grid[2])
    if isinstance(x2_grid, tuple) and len(x2_grid) == 3:
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], x2_grid[2])
    x1_grid = np.asarray(x1_grid).flatten()
    x2_grid = np.asarray(x2_grid).flatten()

    c_levels = np.linspace(0.0, 1.0, c_num_levels)

    # Get regular grid of G x H points, where each point is an (x1, x2) location
    G = x1_grid.size
    H = x2_grid.size    
    x1_GH, x2_GH = np.meshgrid(x1_grid, x2_grid)
    
    # Combine the x1 and x2 values into one array
    # Flattened into M = G x H rows
    # Each row of x_M2 is a 2D vector [x_m1, x_m2]
    x_M2 = np.hstack([x1_GH.flatten()[:,np.newaxis], x2_GH.flatten()[:,np.newaxis]])
    
    # Predict proba for each point in the flattened grid
    yproba1_M = clf.predict_proba(x_M2)[:,1]
    # Reshape the M probas into the GxH 2D field
    yproba1_GH = np.reshape(yproba1_M, x1_GH.shape)
    # Contour plot
    cmap = plt.cm.RdYlBu
    my_contourf_h = plt.contourf(x1_GH, x2_GH, yproba1_GH, levels=c_levels, vmin=0, vmax=1.0, cmap=cmap)
    # Edit the ticks observed
    if x1_ticks is not None:
        plt.xticks(x1_ticks, x1_ticks);
    if x2_ticks is not None:
        plt.yticks(x2_ticks, x2_ticks);
    if do_show_colorbar:
        left, bottom, width, height = plt.gca().get_position().bounds
        cax = plt.gcf().add_axes([left + 1.1*width, bottom, 0.03, height])
        plt.colorbar(my_contourf_h, orientation='vertical', cax=cax, ticks=c_ticks);
        plt.sca(cur_ax);