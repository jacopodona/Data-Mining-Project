import math
import random
import os
import time

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import utils
from get_similarity_scores import get_query_similarity_scores
from baseline import getUserSimilarityMatrix
from input_utils import read_queries, build_queries_on_table, get_queries_set, read_user_list, read_utility_matrix, \
    read_table,get_queries_list

def top_K_highest_values(row, k):
    # get the indices and values of the top five highest values in the array
    top_five_indices = row.argsort()[-k:][::-1]
    top_five_values = row[top_five_indices]
    return top_five_indices, top_five_values

def fillUtilityMatrix(utility_matrix, user_similarity_matrix,query_similarity_matrix, topK):
    # Using a Knn style approach for filling the utility matrix behaviour
    header = utility_matrix.columns.values.tolist()
    original_matrix = utility_matrix.to_numpy()
    user_similarity_matrix=user_similarity_matrix.to_numpy()
    query_similarity_matrix=query_similarity_matrix.to_numpy()
    filled_matrix = utility_matrix.to_numpy()
    for i in tqdm(range(0, len(original_matrix))):
        row = original_matrix[i]
        for j in range(len(row)):
            rating = row[j]
            if math.isnan(rating):  # Update only empty entries in utility matrix
                # Since the maximum similarity of an user is with himself, we get k+1 top values and discard the maximum
                #Get user neighbors and weights
                k_user_indices, k_user_weights = top_K_highest_values(user_similarity_matrix[i], k=topK + 1)
                k_user_indices = k_user_indices[1:]
                k_user_weights = k_user_weights[1:]

                #Get query neighbors and weights
                k_query_indices, k_query_weights = top_K_highest_values(query_similarity_matrix[j], k=topK + 1)
                k_query_indices = k_query_indices[1:]
                k_query_weights = k_query_weights[1:]

                user_values = original_matrix[k_user_indices, j]
                query_values= original_matrix[i, k_query_indices]
                prediction = 0
                weights_sum = 0
                for h in range(0, len(user_values)):
                    v = user_values[h]
                    q = query_values[h]
                    if not math.isnan(v):
                        prediction += v * k_user_weights[h]
                        weights_sum += k_user_weights[h]
                    if not math.isnan(q):
                        prediction += q * k_query_weights[h]*5
                        weights_sum += k_query_weights[h]*5
                try:
                    prediction = int(prediction / weights_sum)
                except (ZeroDivisionError, ValueError) as e:
                    prediction = random.randint(1, 100)
                row[j] = prediction
        filled_matrix[i] = row

    return filled_matrix

def getQuerySimilarityMatrix(queries_file,similarity_scores):
    queries=get_queries_list(queries_file)
    queries_ids = [query[0] for query in queries]
    similarity_matrix=np.zeros((len(queries_ids),len(queries_ids)))
    for i in range(0,len(similarity_matrix)):
        q_id=queries_ids[i]
        row=list(similarity_scores[q_id].values())
        if(i==len(queries_ids)-1 and len(row)!=len(queries_ids)):
            row.append(1)#Bug in last row, last element contains one less value, attach the last remaining value that is the correlation between last item and itself
        similarity_matrix[i]=row
    dataframe=pd.DataFrame(similarity_matrix,columns=queries_ids)
    return dataframe

if __name__ == '__main__':
    start_time=time.time()
    print('Preparing dataset...')
    table = '../DatasetGeneration/tables/people.csv'
    matrix='../DatasetGeneration/utility_matrices/people_utility_matrix.csv'
    queries_file='people.csv'

    table = read_table('people.csv')

    queries = read_queries(queries_file)
    queries = build_queries_on_table(table, queries)

    user_list = read_user_list('people_user_list')

    utility_matrix = read_utility_matrix('people_utility_matrix.csv')

    val_split_percentage=0.8

    #Use section of matrix for training and section for validation
    _, val_full_df = utils.get_train_val_split(utility_matrix, val_split_percentage)
    val_masked_df = utils.mask_val_split(val_full_df, 0.7)
    train_utility_matrix = utils.replaceMatrixSection(utility_matrix, val_masked_df)
    print("Building user similarity matrix...")
    user_similarity=getUserSimilarityMatrix(train_utility_matrix)
    print("Done")
    print("Generating query similarity matrix...")
    query_scores=get_query_similarity_scores(queries)
    query_similarity=getQuerySimilarityMatrix(queries_file,query_scores)
    print("Done")

    k=5 #Pick from [1,2,5,10]
    filled_matrix=fillUtilityMatrix(train_utility_matrix,user_similarity,query_similarity,topK=k)
    _, val_prediction_split = utils.get_train_val_split(filled_matrix, val_split_percentage)
    print('='*40)
    print('K=',k)
    print('Prediction error per item is:',utils.evaluateMAE(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split))
    print('Accuracy on prediction=:',utils.evaluateAccuracy(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split))

    print('Execution time:',round(time.time()-start_time,2))