import csv
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import pandas as pd


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


def replaceMatrixSection(full_df, new_section_df):
    header = full_df.columns.values.tolist()
    new_utility = full_df.copy()
    new_utility = new_utility.to_numpy()
    new_section_df = new_section_df.to_numpy()
    num_rows = len(new_section_df)

    new_utility[-num_rows:, :] = new_section_df
    return pd.DataFrame(new_utility, columns=header)


def get_train_val_split(table, train_user_percentage=0.7):
    """

    :param table: pandas dataframe containing utility matrix
    :param train_user_percentage: Percentage of users in training set with full available utility matrix
    :return: train_dataframe, val_dataframe
    """
    train_split = table[:int(len(table) * train_user_percentage)]
    test_split = table[int(len(table) * train_user_percentage):]
    return train_split, test_split


def mask_val_split(df, query_split_percentage=0.5):
    """
    portion of utility matrix of a fixed percentage of queries
    :param df:
    :param query_split_percentage:
    :return:
    """
    header = df.columns.values.tolist()

    start_column_number = len(df.columns)
    transposed = df.T
    visible_split = transposed[:int(len(transposed) * query_split_percentage)]
    masked_split = transposed[int(len(transposed) * query_split_percentage):]
    visible_split = visible_split.T
    masked_split = masked_split.T
    numpy_table = visible_split.to_numpy()
    final_table = np.empty(shape=np.shape(df.to_numpy()))
    for i in range(0, len(numpy_table)):
        row = numpy_table[i]
        while (len(row) != start_column_number):
            row = np.append(row, float('nan'))
        final_table[i] = row
    masked_df = pd.DataFrame(final_table, columns=header)

    return masked_df

def evaluateMAE(gt_df,masked_df,proposal_df):
    error=0
    gt_df=gt_df.to_numpy()
    masked_df=masked_df.to_numpy()
    if isinstance(proposal_df, pd.DataFrame):
        proposal_df = proposal_df.to_numpy()
    for i in range(0, len(gt_df)):
        row_error = 0
        gt_row = gt_df[i]
        masked_row = masked_df[i]
        pr_row = proposal_df[i]
        rated_items = 0
        gt=[]
        proposals=[]
        for j in range(len(gt_row)):
            if not math.isnan(gt_row[j]) and gt_row[j] != masked_row[j]:  # Compute error only on ground truth values that have been masked
                gt.append(gt_row[j])
                proposals.append(pr_row[j])
                rated_items += 1
        mae = mean_absolute_error(gt, proposals)
        error += mae  # Compute average item error
    error = error / len(gt_df)  # Compute error per row
    return error

def evaluateAccuracy(gt_df,masked_df,proposal_df,huber_threshold=1):
    error=0
    gt_df=gt_df.to_numpy()
    masked_df=masked_df.to_numpy()
    if isinstance(proposal_df, pd.DataFrame):
        proposal_df = proposal_df.to_numpy()
    for i in range(0,len(gt_df)):
        good_predicted_item=0
        bad_predicted_item=0
        rated_items=0
        gt_row=gt_df[i]
        masked_row = masked_df[i]
        pr_row=proposal_df[i]
        user_average_vote=np.nanmean(gt_row)
        for j in range(len(gt_row)):
            if(not math.isnan(gt_row[j]) and gt_row[j]!=masked_row[j]):# Compute error only on ground truth values that have been masked
                #gt = gt_row[j]
                #prediction = pr_row[j]
                user_average_vote=50
                if((gt_row[j]>user_average_vote and pr_row[j]>user_average_vote) or (gt_row[j]<user_average_vote and pr_row[j]<user_average_vote)):
                    good_predicted_item+=1
                else:
                    bad_predicted_item+=1
                rated_items += 1
        try:
            error+=(good_predicted_item/rated_items) #Compute average item error
        except ZeroDivisionError:
            error+=0
    error=error/len(gt_df) #Compute error per row
    return error

def evaluateRMSE(gt_df,masked_df,proposal_df):
    error = 0
    gt_df = gt_df.to_numpy()
    masked_df = masked_df.to_numpy()
    if isinstance(proposal_df, pd.DataFrame):
        proposal_df = proposal_df.to_numpy()
    for i in range(0, len(gt_df)):
        row_error = 0
        gt_row = gt_df[i]
        masked_row = masked_df[i]
        pr_row = proposal_df[i]
        rated_items = 0
        gt=[]
        proposals=[]
        for j in range(len(gt_row)):
            if not math.isnan(gt_row[j]) and gt_row[j] != masked_row[j]:  # Compute error only on ground truth values that have been masked
                # through SVD
                gt.append(gt_row[j])
                proposals.append(pr_row[j])
                rated_items += 1
        rmse=np.sqrt(mean_squared_error(gt,proposals))
        error += rmse  # Compute average item error
    error = error / len(gt_df)  # Compute error per row
    return error


def convertNaN(utility_matrix):
    header = utility_matrix.columns.values.tolist()
    # table=utility_matrix.to_numpy()
    for i in range(0, len(header)):
        for j in range(0, len(utility_matrix)):
            if math.isnan(utility_matrix.iat[j, i]):
                utility_matrix.iat[j, i] = 0
    # matrix=pd.DataFrame(utility_matrix, columns = header)
    return utility_matrix
