import numpy as np
from autograd import grad, value_and_grad
import os
import sys
import pandas as pd

class AbstractBaseCollabFilterSGD(object):
    """ Base class for user-movie rating prediction via matrix factorization.

    Attributes set by calling __init__()
    ------------------------------------
    step_size  : float
    n_epochs   : int
    batch_size : int
    n_factors  : int [optional]
    alpha      : float [optional]

    Attributes set after calling init_param_dict() and updated by fit()
    -------------------------------------------------------------------
    param_dict : dict
        Written generically in this base class, each subclass should override
        Keys are string names of parameters
        Values are *numpy arrays* of parameter values
    """

    def __init__(self,
            step_size=0.1, n_epochs=100, batch_size=1000,
            n_factors=0, alpha=0.00, random_state=20190415):
        """ Construct instance and set its attributes

        Args
        ----
        step_size  : float
            Step size / learning rate used in each gradient descent step.
        n_epochs : int
            Total number of epochs (complete passes thru provided training set)
            to complete during a call to fit. 
        batch_size : int
            Number of rating examples to process in each 'batch' or 'minibatch'
            of stochastic gradient descent. 
        n_factors : int
            Number of dimensions each per-user/per-item vector has.
            (Will be unused by simpler models).
        alpha : float
            Regularization strength (must be >= 0.0).

        Returns
        -------
        New instance of this class
        """
        self.n_factors  = int(n_factors)
        self.alpha      = float(alpha)
        self.step_size  = step_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state


    def evaluate_perf_metrics(self, user_id_N, item_id_N, ratings_N):
        ''' Evaluate performance metrics for current model on given dataset.

        Returns
        -------
        perf_dict : dict
            Key,value pairs represent the names and values of useful metrics.
        '''
        n_examples = user_id_N.size
        yhat_N = self.predict(user_id_N, item_id_N, **self.param_dict)
        mse = np.mean(np.square(yhat_N - ratings_N))
        mae = np.mean(np.abs(yhat_N - ratings_N))
        return dict(mse=mse, mae=mae)

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Template method to compute loss at specific parameters.
        '''
        raise NotImplementedError("Subclasses need to override this method")

    def calc_loss_and_grad_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss and gradient at specific parameters.

        Uses autograd package to compute gradients.

        Subclasses should NOT need to override this in general, as long as
        the method `calc_loss_wrt_parameter_dict` is written correctly.

        Returns
        -------
        loss : scalar
        grad_dict : dict
            Keys are string names of parameters.
            Values are NumPy arrays, providing gradient of the parameter.
            Will have exactly the same keys as a valid param_dict
        '''
        try:
            self._calc_loss_and_grad_wrt_param_dict
        except AttributeError:
            self._calc_loss_and_grad_wrt_param_dict = value_and_grad(
                self.calc_loss_wrt_parameter_dict, argnum=[0])
        loss, grad_dict_tuple = self._calc_loss_and_grad_wrt_param_dict(
            self.param_dict, data_tuple)
        grad_dict = grad_dict_tuple[0] # Unpack tuple output of autograd
        return loss, grad_dict

    def fit(self, train_data_tuple, valid_data_tuple=None):
        """ Fit latent factor model to user-movie ratings via gradient descent.

        Calling this method will attempt to solve the optimization problem:

            U^*, V^* = min_{U, V} loss_total(r, U, V)

        given a dataset of N user-item ratings r_{i,j} for user i and item j.

        The loss has two terms, the error and regularization penalty:

            loss_total(r, U, V) = error(r, U, V) + \alpha * penalty(U, V)

        The regression error term is just squared error over observed ratings:
        
            error(r, U, V) = \sum_{i,j} ( r_i,j - dot(U[i], V[j]) )^2
        
        And the regularization penalty is:

            penalty(U, V) = \sum_i L2norm(U_i) + \sum_j L2norm(V_j)

        Args
        ----
        train_data_tuple : length-3 tuple
            Looks like (user_id, item_id, rating)
        valid_data_tuple : length-3 tuple
            Looks like (user_id, item_id, rating)

        Returns
        -------
        None.

        Post Condition
        --------------
        Internal `param_dict` attribute updated.
        """
        n_total = train_data_tuple[0].size
        batch_loader = RatingsMiniBatchIterator(
            *train_data_tuple,
            batch_size=self.batch_size,
            random_state=self.random_state)

        self.trace_epoch = []
        self.trace_loss = []
        self.trace_smooth_loss = []
        self.trace_mae_train = []
        self.trace_mae_valid = []

        self.all_loss = []

        ## Store list of L1 gradient norms for each parameter
        self.trace_norm_per_param = dict()
        for key in self.param_dict.keys():
            self.trace_norm_per_param[key] = list()
        self.trace_smooth_norm_per_param = dict()
        for key in self.param_dict.keys():
            self.trace_smooth_norm_per_param[key] = list()

        for epoch_count in range(self.n_epochs): 
            epoch = 1.0 * epoch_count
            batch_loader.shuffle()

            for i, batch_tuple in enumerate(batch_loader):

                ## Compute loss and gradient
                # loss : scalar float
                # grad_dict : dict
                #   Keys are string names of individual parameters
                #   Values are autograd-generated numpy arrays
                loss, grad_dict = self.calc_loss_and_grad_wrt_parameter_dict(
                    self.param_dict, batch_tuple)

                ## Rescale loss and gradient vectors
                # So we always estimate the *per-example loss*
                n_per_batch = batch_tuple[0].size
                scale = 1.0 / n_per_batch
                loss *= scale
                for key, arr in grad_dict.items():
                    arr *= scale
                self.all_loss.append(loss)

                ## Periodically report progress to stdout
                ## & write to internal state attributes: self.trace_*
                do_report_now = self.check_if_report_progress_now(
                    epoch_count, self.n_epochs, i, batch_loader.n_batches)
                if do_report_now:
                    self.trace_epoch.append(epoch)
                    self.trace_loss.append(loss)

                    # Compute MAE/MSE metrics on training and validation data
                    train_perf_dict = self.evaluate_perf_metrics(*train_data_tuple)
                    valid_perf_dict = self.evaluate_perf_metrics(*valid_data_tuple)
                    self.trace_mae_train.append(train_perf_dict['mae'])
                    self.trace_mae_valid.append(valid_perf_dict['mae'])

                    # Compute 'smoothed' loss by averaging over last B batches
                    # Might remove some of the stochasticity in using only the
                    # loss from most recent batch.
                    smooth_loss = np.mean(self.all_loss[-batch_loader.n_batches:])
                    self.trace_smooth_loss.append(smooth_loss)

                    # Compute L1 norm of gradient of each parameter
                    avg_grad_norm_str_list = []
                    for key, arr in grad_dict.items():
                        norm = np.mean(np.abs(arr))
                        self.trace_norm_per_param[key].append(norm)
                        cur_norm_str = "grad_wrt_%s %11.5f" % (key, norm)
                        avg_grad_norm_str_list.append(cur_norm_str)
                    avg_grad_norm_str = ' | '.join(avg_grad_norm_str_list)

                    print("epoch %11.3f | loss_total % 11.5f | train_MAE % 11.5f | valid_MAE % 11.5f | %s" % (
                        epoch, loss if epoch <= 2 else smooth_loss,
                        train_perf_dict['mae'], valid_perf_dict['mae'],
                        avg_grad_norm_str))

                ## Update each parameter by taking step in direction of gradient
                epoch += n_per_batch / n_total 
                for key, arr in self.param_dict.items():
                    arr[:] = arr - self.step_size * grad_dict[key]

        # That's all folks.


    def check_if_report_progress_now(
            self, epoch_count, max_epoch,
            batch_count_within_epoch, max_batch_per_epoch):
        ''' Helper method to decide when to report progress on valid set.

        Will check current training progress (num steps completed, etc.)
        and determine if we should perform validation set diagnostics now.

        Returns
        -------
        do_report_now : boolean
            True if report should be done, False otherwise
        '''
        is_last_step = (
            epoch_count == (max_epoch - 1)
            and batch_count_within_epoch == (max_batch_per_epoch - 1))

        if epoch_count == 0 and batch_count_within_epoch < 4:
            # Do a report at each of first 4 steps
            return True
        elif is_last_step:
            # Do a report at final step
            return True

        for max_epoch, freq in [
                (2, 1/8),
                (8, 1/2),
                (32, 1.0),
                (128, 2.0),
                (512, 4.0),
                (2048, 8.0)]:
            if epoch_count >= max_epoch:
                continue
            if freq < 1:
                cur_counts = np.arange(max_batch_per_epoch)
                ideal_counts = np.arange(0, 1, freq) * float(max_batch_per_epoch)
                report_counts = np.unique(np.searchsorted(cur_counts, ideal_counts))
                if batch_count_within_epoch in report_counts:
                    return True
            else:
                if epoch_count % freq == 0 and batch_count_within_epoch == 0:
                    return True
        return False




class RatingsMiniBatchIterator(object):
    """ Iterator to loop through small batches of (user,item,rating) examples

    Given arrays defining (i, j, k) values,
    will produce minibatches of these values of desired batch size.

    Final batch may be (much) smaller than desired batch size.

    Usage
    -----
    >>> x = np.arange(7)
    >>> y = np.arange(7)
    >>> z = np.arange(7)
    >>> batch_loader = RatingsMiniBatchIterator(
    ...     x, y, z, batch_size=3, random_state=8675309)
    >>> for batch in batch_loader:
    ...     print(batch)
    (array([0, 1, 2]), array([0, 1, 2]), array([0, 1, 2]))
    (array([3, 4, 5]), array([3, 4, 5]), array([3, 4, 5]))
    (array([6]), array([6]), array([6]))

    # Shuffle and show another epoch
    >>> batch_loader.shuffle()
    >>> for batch in batch_loader:
    ...     print(batch)
    (array([6, 2, 0]), array([6, 2, 0]), array([6, 2, 0]))
    (array([3, 5, 4]), array([3, 5, 4]), array([3, 5, 4]))
    (array([1]), array([1]), array([1]))

    # Shuffle and show another epoch
    >>> batch_loader.shuffle()
    >>> for batch in batch_loader:
    ...     print(batch)
    (array([3, 1, 6]), array([3, 1, 6]), array([3, 1, 6]))
    (array([4, 5, 0]), array([4, 5, 0]), array([4, 5, 0]))
    (array([2]), array([2]), array([2]))
    """

    def __init__(self, us, vs, ratings, random_state=np.random, batch_size=64):
        ''' Construct iterator and set its attributes

        Args
        ----
        us : int array
        vs : int array
        ratings : int array
        batch_size : int
            
        Returns
        -------
        New instance of this class 
        '''
        try:
            self.random_state = np.random.RandomState(random_state)
        except Exception:
            self.random_state = random_state
        self.u = us
        self.v = vs
        self.rating = ratings
        self.batch_size = batch_size
        self.n_examples = np.size(us, axis=0)
        self.n_batches = int(np.ceil(self.n_examples / batch_size))

        self.batch_size_B = batch_size * np.ones(self.n_batches)

        # Final batch might be a bit smaller than others
        remainder = int(self.n_examples - np.sum(self.batch_size_B[:-1]))
        self.batch_size_B[-1] = remainder
        assert np.sum(self.batch_size_B) == self.n_examples

        # Set current batch counter to 0
        self.cur_batch_id = 0


    def shuffle(self, random_state=None):
        """ Shuffle internal dataset to a random order

        Returns
        -------
        Nothing.
        """        
        if random_state is None:
            random_state = self.random_state
        perm_ids = random_state.permutation(self.n_examples)
        self.u = self.u[perm_ids]
        self.v = self.v[perm_ids]
        self.rating = self.rating[perm_ids]


    def __next__(self):
        """ Get next batch of ratings data

        Returns
        -------
        u : 1D array of int
            User ids
        v : 1D array of int
            Item ids
        ratings: 1D array of int
            Rating values
        """
        if self.cur_batch_id >= self.n_batches:
            self.cur_batch_id = 0
            raise StopIteration
        else:
            start = int(np.sum(self.batch_size_B[:self.cur_batch_id]))
            stop = start + int(self.batch_size_B[self.cur_batch_id])
            cur_batch_tuple = (
                self.u[start:stop],
                self.v[start:stop],
                self.rating[start:stop])
            self.cur_batch_id += 1
            return cur_batch_tuple

    def __iter__(self):
        ''' Allow using this object directly as an iterator

        That is, we can use syntax like:
        
        for batch in RatingsMiniBatchIterator(...):
            do something

        This method tells python that this object supports this.
        '''
        return self