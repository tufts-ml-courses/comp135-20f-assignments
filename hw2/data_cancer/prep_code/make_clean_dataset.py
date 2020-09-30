"""
Name        Data Type   Meas.   Description
----        ---------   -----   -----------
"""

import pandas as pd
import numpy as np
import sklearn.model_selection
    
y_col_names = [
    'cancer',
    ]
x_col_names = ['age', 'famhistory', 'marker']

if __name__ == '__main__':
    n_test = 180
    csv_df = pd.read_csv('../raw_data/dca.csv', delimiter=',')

    # Keep only relevant data columns, reorder so it goes x then y
    keep_df = csv_df[x_col_names + y_col_names].copy()

    ## Scramble all examples then divide into train/validation/test
    rest_df, test_df  = sklearn.model_selection.train_test_split(
        keep_df, test_size=n_test, random_state=42, shuffle=True, stratify=keep_df[y_col_names[0]])
    train_df, valid_df = sklearn.model_selection.train_test_split(
        rest_df, test_size=n_test, random_state=42, shuffle=True, stratify=rest_df[y_col_names[0]])

    key_var_names = [
        'cancer',
        'age',
        'marker',
        ]

    def pprint_histogram(arr, label_str, bin_edges):
        counts, edges = np.histogram(arr, bin_edges)
        fracs = np.asarray(counts, dtype=np.float32) / arr.size
        print(label_str)
        print('   ' + ' '.join(['% 6.1f-% 6.1f' % (
            bin_edges[b], bin_edges[b+1])
            for b in range(counts.size)]))
        print('   ' + ' '.join(['%13d' % c for c in counts]))
        print('   ' + ' '.join(['%13.2f' % f for f in fracs]))

    for var_name in key_var_names:
        bin_edges = np.linspace(
            np.percentile(keep_df[var_name], 2),
            np.percentile(keep_df[var_name], 98),
            10
            )
        print('')
        print('====== %s' % var_name)
        pprint_histogram(train_df[var_name], 'train', bin_edges)
        pprint_histogram(valid_df[var_name], 'valid', bin_edges)
        pprint_histogram(test_df[var_name], 'test', bin_edges)


    train_df.to_csv('../x_train.csv', index=False, columns=x_col_names)
    valid_df.to_csv('../x_valid.csv', index=False, columns=x_col_names)
    test_df.to_csv('../x_test.csv', index=False, columns=x_col_names)

    train_df.to_csv('../y_train.csv', index=False, columns=y_col_names)
    valid_df.to_csv('../y_valid.csv', index=False, columns=y_col_names)
    test_df.to_csv('../y_test.csv', index=False, columns=y_col_names)


