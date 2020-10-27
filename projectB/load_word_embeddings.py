import pandas as pd
import numpy as np
import os

import sklearn.neighbors

from collections import OrderedDict

if __name__ == '__main__':

    zip_file_path = os.path.join(
        'pretrained_embedding_vectors/',
        'glove.6B.50d.txt.zip')

    word_embeddings = pd.read_csv(
        zip_file_path,
        header=None, sep=' ', index_col=0,
        nrows=100000, compression='zip', encoding='utf-8', quoting=3)

    # Build a dict that will map from string word to 50-dim vector
    word_list = word_embeddings.index.values.tolist()
    word2vec = OrderedDict(zip(word_list, word_embeddings.values))

    # Show some examples of word embeddings
    # Each word will get mapped to a 
    n_words = len(word2vec.keys())

    print("Loaded pretrained embeddings for %d possible words" % n_words)
    print("Each embedding vector has %d dimensions" % (
        list(word2vec.values())[0].size))

    print("word2vec['london'] = ")
    print(word2vec['london'])

    print("word2vec['england'] = ")
    print(word2vec['england'])

    # Try some analogies (just for fun)
    def analogy_lookup(a1, a2, b1):
        target_vec = word2vec[a2] - word2vec[a1] + word2vec[b1]
        knn = sklearn.neighbors.NearestNeighbors(n_neighbors=7, metric='euclidean', algorithm='brute')
        knn.fit(word_embeddings.values)
        dists, indices = knn.kneighbors(target_vec[np.newaxis,:])
        print("")
        print("Query:  %s is to %s   as   %s is to ____" % (a1, a2, b1))
        print("Best answers (ranked by distance in vector space)")
        for ii, vv in enumerate(indices[0]):
            print("   %20s  at dist %.3f" % (word_list[vv], dists[0,ii]))

    analogy_lookup('england', 'london', 'france')
    analogy_lookup('england', 'london', 'germany')
    analogy_lookup('england', 'london', 'japan')
    analogy_lookup('england', 'london', 'indonesia')

    analogy_lookup('swim', 'swimming', 'run')
