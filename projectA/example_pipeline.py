'''
Short examples of how to design custom pipelines that
concatenate multiple feature transformations.

See Also
--------
See also the day04 lab on Pipelines in sklearn.

See also the sklearn documentation on pipelines.
'''

import numpy as np

from matplotlib import pyplot as plt

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class SquaredFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self):
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        """ Average all feature values into a new feature column

        Args
        ----
        x : 2D array, size F

        Returns
        -------
        feat : 2D array, size N x F
            One feature extracted for each example
        """
        return np.square(x)

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['square_of_%02d' % f for f in range(x.shape[1])]
        return self

class AverageValueFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to the *sum* of all pixels in image
    """

    def __init__(self):
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        """ Average all feature values into a new feature column

        Returns
        -------
        feat : 2D array, size N x 1
            One feature extracted for each example
        """
        return np.sum(x, axis=1)[:,np.newaxis]

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['avg_of_%s-%s' % (0, x.shape[1])]
        return self

if __name__ == '__main__':
   
    # Create 2 example images in 8x8 pixel space
    N = 2
    F = 64

    one_88 = np.asarray([ 
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0., 0., 0.,],
        ])

    eight_88 = np.asarray([ 
        [0., 0., 1., 1., 1., 1., 0., 0.,],
        [0., 1., 1., 0., 0., 1., 1., 0.,],
        [0., 1., 1., 0., 0., 1., 1., 0.,],
        [0., 0., 1., 1., 1., 1., 0., 0.,],
        [0., 0., 1., 1., 1., 1., 0., 0.,],
        [0., 1., 1., 0., 0., 1., 1., 0.,],
        [0., 1., 1., 0., 0., 1., 1., 0.,],
        [0., 0., 1., 1., 1., 1., 0., 0.,],
        ])

    orig_feat_names = ['pixel%02d' % f for f in range(F)]
    x_NF = np.vstack([
        one_88.reshape(1, F),
        eight_88.reshape(1, F)
        ])

    y_N = np.asarray([0, 1])

    feature_tfmr = sklearn.pipeline.FeatureUnion(transformer_list=[
            ('orig', sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
            ('sq', SquaredFeatureExtractor()),
            ('av', AverageValueFeatureExtractor()),
            ])
    classifier = sklearn.linear_model.LogisticRegression(C=1.0)

    pipeline = sklearn.pipeline.Pipeline([
        ('step1', feature_tfmr),
        ('step2', classifier)
        ])
    pipeline.fit(x_NF, y_N)

    phi_NG = pipeline.named_steps['step1'].transform(x_NF)


    print("ARRAYS")
    print("Raw features: shape %s" % str(x_NF.shape))
    print(x_NF)
    print("Transformed feature array phi_NG: shape %s" % str(phi_NG.shape))
    print(phi_NG)

    print("NAMES")
    phi_feat_names = pipeline.named_steps['step1'].get_feature_names()
    print("Raw features fed into the pipeline:")
    print(orig_feat_names)
    print("Transformed features produced by the pipeline:")
    print(phi_feat_names)



