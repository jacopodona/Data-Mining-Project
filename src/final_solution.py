import math
import random
# import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import scipy
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import utils
from get_similarity_scores import get_query_similarity_scores
from input_utils import read_queries, build_queries_on_table, read_utility_matrix, read_table, get_queries_list


def getUserSimilarityMatrix(utility_matrix: DataFrame) -> DataFrame:
    ids = utility_matrix.index.tolist()
    table = utility_matrix.to_numpy()
    # Get a "normalized" utility matrix, where every item rating equals to the difference between the actual rating
    # in the 1-100 range minus the user average grade
    row_avg = np.nanmean(table, axis=1)  # Ignores NaN when computing mean
    normalized_matrix = table - row_avg[:, np.newaxis]
    # Replaces NaN values with 0s
    normalized_matrix = np.nan_to_num(normalized_matrix)
    matrix = scipy.sparse.csr_matrix(normalized_matrix)  # pandas DF into scipy sparse matrix

    # Compute user similarity matrix using cosine similarity
    similarities = cosine_similarity(matrix)
    similarities = pd.DataFrame(similarities, index=ids, columns=ids)
    return similarities


def top_K_highest_values(row: ndarray, k: int) -> (ndarray, ndarray):
    # get the indices and values of the top five highest values in the array
    top_five_indices = row.argsort()[-k:][::-1]
    top_five_values = row[top_five_indices]
    normalized_top_five_values = top_five_values / np.linalg.norm(top_five_values)
    return top_five_indices, normalized_top_five_values


def fill_utility_matrix(utility_matrix: DataFrame,
                        user_similarity_matrix: DataFrame,
                        query_similarity_matrix: DataFrame,
                        top_k: int) -> ndarray:
    # Using a Knn style approach for filling the utility matrix behaviour
    # header = utility_matrix.columns.values.tolist()
    original_matrix = utility_matrix.to_numpy()
    user_similarity_matrix = user_similarity_matrix.to_numpy()
    query_similarity_matrix = query_similarity_matrix.to_numpy()
    filled_matrix = utility_matrix.to_numpy()
    for i in tqdm(range(0, len(original_matrix))):
        row = original_matrix[i]
        for j in range(len(row)):
            rating = row[j]
            if math.isnan(rating):  # Update only empty entries in utility matrix
                # Since the maximum similarity of a user is with himself, we get k+1 top values and discard the maximum
                # Get user neighbors and weights
                k_user_indices, k_user_weights = top_K_highest_values(user_similarity_matrix[i], k=top_k + 1)
                k_user_indices = k_user_indices[1:]
                k_user_weights = k_user_weights[1:]
                k_user_weights = k_user_weights * 3

                # Get query neighbors and weights
                k_query_indices, k_query_weights = top_K_highest_values(query_similarity_matrix[j], k=top_k + 1)
                k_query_indices = k_query_indices[1:]
                k_query_weights = k_query_weights[1:]

                user_values = original_matrix[k_user_indices, j]
                query_values = original_matrix[i, k_query_indices]
                prediction = 0
                weights_sum = 0
                for h in range(0, len(user_values)):
                    v = user_values[h]
                    q = query_values[h]
                    if not math.isnan(v):
                        prediction += v * k_user_weights[h]
                        weights_sum += k_user_weights[h]
                    if not math.isnan(q):
                        prediction += q * k_query_weights[h] * 5
                        weights_sum += k_query_weights[h] * 5
                try:
                    prediction = int(prediction / weights_sum)
                except (ZeroDivisionError, ValueError) as e:
                    prediction = random.randint(1, 100)
                row[j] = prediction
        filled_matrix[i] = row

    return filled_matrix


def get_query_similarity_matrix(queries_file: str, similarity_scores: Dict) -> DataFrame:
    queries = get_queries_list(queries_file)
    queries_ids = [query[0] for query in queries]
    similarity_matrix = np.zeros((len(queries_ids), len(queries_ids)))
    for i in range(0, len(similarity_matrix)):
        q_id = queries_ids[i]
        row = list(similarity_scores[q_id].values())
        if i == len(queries_ids) - 1 and len(row) != len(queries_ids):
            row.append(1)
            # Bug in last row, last element contains one less value,
            # attach the last remaining value that is the correlation
            # between last item and itself
        similarity_matrix[i] = row
    dataframe = pd.DataFrame(similarity_matrix, columns=queries_ids)
    return dataframe


if __name__ == '__main__':
    start_time = time.time()
    print('Preparing dataset...')
    # table = '../DatasetGeneration/tables/movies.csv'
    # matrix = '../DatasetGeneration/utility_matrices/movies_utility_matrix.csv'
    domain = "people"
    queries_file = f'{domain}.csv'

    table = read_table(f'{domain}.csv')

    queries = read_queries(queries_file)
    queries = build_queries_on_table(table, queries)

    utility_matrix = read_utility_matrix(f'{domain}_utility_matrix.csv')

    val_split_percentage = 0.8

    # Use section of matrix for training and section for validation
    _, val_full_df = utils.get_train_val_split(utility_matrix, val_split_percentage)
    val_masked_df = utils.mask_val_split(val_full_df, 0.7)
    train_utility_matrix = utils.replaceMatrixSection(utility_matrix, val_masked_df)

    print("Building user similarity matrix...")
    user_similarity = getUserSimilarityMatrix(train_utility_matrix)
    print("Done")

    print("Generating query similarity matrix...")
    query_scores = get_query_similarity_scores(queries)
    query_similarity = get_query_similarity_matrix(queries_file, query_scores)
    print("Done")

    k = 20  # Pick from [1,2,5,10]
    filled_matrix = fill_utility_matrix(train_utility_matrix, user_similarity, query_similarity, top_k=k)
    _, val_prediction_split = utils.get_train_val_split(filled_matrix, val_split_percentage)
    utils.print_results(k, val_full_df, val_masked_df, val_prediction_split, start_time)
