import math
import random

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


import utils


def getUserSimilarityMatrix(utility_matrix):
    ids = utility_matrix.index.tolist()
    table = utility_matrix.to_numpy()
    # Replaces NaN values with 0s
    matrix = np.nan_to_num(table)
    matrix = scipy.sparse.csr_matrix(matrix)  # pandas DF into scipy sparse matrix

    # Compute user similarity matrix using cosine similarity
    similarities = cosine_similarity(matrix)
    similarities = pd.DataFrame(similarities, index=ids, columns=ids)
    return similarities


def top_K_highest_values(row, k):
    # get the indices and values of the top five highest values in the array
    top_five_indices = row.argsort()[-k:][::-1]
    top_five_values = row[top_five_indices]
    return top_five_indices, top_five_values


def fillUtilityMatrix(utility_matrix, similarity_matrix, topK):
    # Using a Knn style approach for filling the utility matrix behaviour
    header = utility_matrix.columns.values.tolist()
    original_matrix = utility_matrix.to_numpy()
    similarity_matrix=similarity_matrix.to_numpy()
    filled_matrix = utility_matrix.to_numpy()
    for i in tqdm(range(0, len(original_matrix))):
        row = original_matrix[i]
        for j in range(len(row)):
            rating = row[j]
            if math.isnan(rating):  # Update only empty entries in utility matrix
                indices, weights = top_K_highest_values(similarity_matrix[i], k=topK + 1)  # Since the maximum similarity of an user is
                # with himself, we get k+1 top values and discard the maximum
                indices = indices[1:]
                weights = weights[1:]
                values = original_matrix[indices, j]
                prediction = 0
                weights_sum = 0
                for h in range(0, len(values)):
                    v = values[h]
                    if not math.isnan(v):
                        prediction += v * weights[h]
                        weights_sum += weights[h]
                try:
                    prediction = int(prediction / weights_sum)
                except (ZeroDivisionError, ValueError) as e:
                    prediction = random.randint(1, 100)
                row[j] = prediction
        filled_matrix[i] = row

    return filled_matrix


if __name__ == '__main__':
    print('Preparing dataset...')
    table = '../DatasetGeneration/tables/people.csv'
    queries = utils.read_queries('../DatasetGeneration/queries/people.csv')
    matrix='../DatasetGeneration/utility_matrices/people_utility_matrix.csv'

    val_split_percentage=0.8

    #Get tables loaded into memory
    items_table=pd.read_csv(table)
    utility_matrix=pd.read_csv(matrix)

    #Use section of matrix for training and section for validation
    _, val_full_df = utils.get_train_val_split(utility_matrix, val_split_percentage)
    val_masked_df = utils.mask_val_split(val_full_df, 0.7)
    train_utility_matrix = utils.replaceMatrixSection(utility_matrix, val_masked_df)

    user_similarity=getUserSimilarityMatrix(train_utility_matrix)
    utility = utility_matrix.to_numpy()

    k=20 #Pick from [1,2,5,10]
    filled_matrix=fillUtilityMatrix(train_utility_matrix,user_similarity,topK=k)
    _, val_prediction_split = utils.get_train_val_split(filled_matrix, val_split_percentage)
    print('='*40)
    print('K=',k)
    print('MAE per row:',utils.evaluateMAE(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split))
    print('RMSE per row=:',utils.evaluateRMSE(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split))
    print('Accuracy on prediction=:',utils.evaluateAccuracy(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split))
