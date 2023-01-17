import pandas
import os
import csv
import random
import utils
import pandas as pd
import time


def displayDatasetStatistics(df):
    header=df.columns.values.tolist()
    table=df.to_numpy()
    for col in header:
        print(col,len(pd.unique(df[col])))


if __name__ == '__main__':
    table = '../DatasetGeneration/tables/people.csv'
    queries = utils.read_queries('../DatasetGeneration/queries/people.csv')

    df=pd.read_csv(table)
    displayDatasetStatistics(df)