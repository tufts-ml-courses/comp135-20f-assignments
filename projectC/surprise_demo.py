'''
Quick demonstration of how to use surprise with our movie lens 100k dataset

'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate

reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)

## Load the entire dev set in surprise's format
train_set = Dataset.load_from_file(
    'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
train_set = train_set.build_full_trainset()

# Use the SVD algorithm
for n_factors in [1]:
    ## Fit model like our M3
    model = SVD(n_factors=n_factors)
    model.fit(train_set)

    print("global mean:")
    print(model.trainset.global_mean)
    print("shape of bias_per_item: ")
    print(model.bi.shape)
    print("shape of bias_per_user: ")
    print(model.bu.shape)
    print("shape of U (per user vectors): ")
    print(model.pu.shape)
    print("shape of V (per item vectors): ")
    print(model.qi.shape)
