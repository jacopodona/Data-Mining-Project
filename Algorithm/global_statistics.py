import pandas
import os
import csv
import random
import numpy as np
import utils
import pandas as pd
import time


def displayDatasetStatistics(df):
    header=df.columns.values.tolist()
    table=df
    for col in header:
        print(col,len(pd.unique(df[col])))


if __name__ == '__main__':
    table = '../DatasetGeneration/tables/people.csv'
    queries = utils.read_queries('../DatasetGeneration/queries/people.csv')
    matrix='../DatasetGeneration/utility_matrices/people_utility_matrix.csv'

    df=pd.read_csv(table)
    utility_matrix=pd.read_csv(matrix)
    #displayDatasetStatistics(df)
    print('Generating train-val splits')
    train_df,val_full_df=utils.get_train_val_split(utility_matrix,0.8)
    print('Masking splits')
    val_masked_df=utils.mask_val_split(val_full_df,0.5)
    test_table = pd.DataFrame(np.ones(shape=np.shape(val_full_df.to_numpy())),columns=val_masked_df.columns.values.tolist())
    print('Average error is: ',utils.evaluate(val_full_df,test_table))