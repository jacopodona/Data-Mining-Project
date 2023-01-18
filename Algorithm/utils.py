import math
import os
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def read_queries(file_path):
    queries = []
    with open(file_path, 'r') as fp:
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            if row:
                queries.append(row)
    queries = [query[0].split(",") for query in queries]
    queries_ids = [query[0] for query in queries]
    query_ids_len = len(queries_ids)

    return queries

def get_train_val_split(table,train_user_percentage=0.7):
    '''

    :param table: pandas dataframe containing utility matrix
    :param train_user_percentage: Percentage of users in training set with full available utility matrix
    :return: train_dataframe, val_dataframe
    '''
    train_split=table[:int(len(table)*train_user_percentage)]
    test_split=table[int(len(table)*train_user_percentage):]
    return train_split,test_split

def mask_val_split(df,query_split_percentage=0.5):
    '''
    portion of utility matrix of a fixed percentage of queries
    :param df:
    :param query_split_percentage:
    :return:
    '''
    header = df.columns.values.tolist()

    start_column_number=len(df.columns)
    transposed=df.T
    visible_split=transposed[:int(len(transposed)*query_split_percentage)]
    masked_split=transposed[int(len(transposed)*query_split_percentage):]
    visible_split=visible_split.T
    masked_split=masked_split.T
    numpy_table=visible_split.to_numpy()
    final_table=np.empty(shape=np.shape(df.to_numpy()))
    for i in range(0,len(numpy_table)):
        row=numpy_table[i]
        while(len(row)!=start_column_number):
            row=np.append(row,float('nan'))
        final_table[i]=row
    masked_df= pd.DataFrame(final_table, columns = header)

    return masked_df

def evaluate(gt_df,proposal_df,huber_threshold=5):
    error=0
    gt_df=gt_df.to_numpy()
    proposal_df = proposal_df.to_numpy()
    for i in range(0,len(gt_df)):
        row_error=0
        gt_row=gt_df[i]
        pr_row=proposal_df[i]
        for j in range(len(gt_row)):
            if(not math.isnan(gt_row[j])):
                row_error+=abs((gt_row[j]-pr_row[j])) #Element wise MAE
                #row_error+=(gt_row[j]-pr_row[j])**2 #Element wise MSE
                '''delta=abs(gt_row[j]-pr_row[j])   #Huber method
                if(delta<huber_threshold):
                    row_error+=(gt_row[j]-pr_row[j])**2
                else:
                    row_error+=delta'''
        error=row_error/len(gt_df)
    return error



