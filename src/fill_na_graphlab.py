from __future__ import division
__author__ = 'Vladimir Iglovikov'

import numpy as np
# import pandas as pd
import sys
import graphlab as gl

'''
This file will contain functions that will fill missed values in the dataframe
'''

def fill_missed(df, label, verbose=False, summary=True):
    '''

    :param df: dataframe with missed values
    :param label: column to fill
    :param method: classifier or regressor to use
    :return: df with filled na values
    '''

    #split df into df with missing values in the desired column and rest of it

    df_id = range(df.num_rows())

    df["id"] = df_id
    train, test = df.dropna_split()
    features = df.column_names()

    features.remove(label)
    features.remove("id")

    model = gl.boosted_trees_regression.create(train, target=label,
                                               features=features,
                                               # max_iterations=300,
                                               verbose=verbose,
                                               # max_depth=10,
                                               # step_size=0.1
                                               )

    if summary:
        print model.summary()

    prediction = model.predict(test)

    test[label] = prediction

    result = train.append(test)
    result = result.sort("id")
    del result["id"]
    return result

def fill_missed_all(df, verbose=False, summary=True):
    '''

    :param df: dataframe with missed values    
    :return: df with filled na values
    '''
    #create list with tuples (# of missed values, column name)

    list_with_na = []
    to_drop = []
    for column in df.column_names():        
        num_na = sum(result[column].apply(lambda x: np.isnan(x)))
        if num_na > 0:
            list_with_na += [(num_na, column)]
            to_drop += [column]

    list_with_na.sort()

    if list_with_na == []:
        return df

    temp_df = df
    temp_df = temp_df.remove_columns(to_drop)

    for i, column in list_with_na:
        print "filling " + column
        temp_df[column] = df[column]
        temp_df = fill_missed(temp_df, column, verbose=verbose, summary=summary)

    return temp_df.to_dataframe()