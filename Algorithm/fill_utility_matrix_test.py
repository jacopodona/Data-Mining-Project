import argparse
import os
import random
from typing import Set, Dict

import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm

from baseline import getUserSimilarityMatrix
from get_similarity_scores import get_query_similarity_scores, get_query_for_user, get_similar_users
from input_utils import read_queries, build_queries_on_table, get_queries_set, read_user_list, read_utility_matrix, \
    read_table

ASSETS_DIR = "../DatasetGeneration/assets"
UTILITY_MATRICES_DIR = "../DatasetGeneration/utility_matrices"
TABLE_DIR = "../DatasetGeneration/tables"
QUERY_DIR = "../DatasetGeneration/queries"
USER_DIR = "../DatasetGeneration/users"


def compute_prediction_on_user(utility_matrix: DataFrame, query: str, has_users: bool, similar_users: Series) -> float:
    """
    Predict the rating based on the ratings for that query of the 5 most similar users, if there are no other
    similar users that have rated that query return 0.0
    :param utility_matrix:
    :param query:
    :param has_users: if there are similar users that have rated that query
    :param similar_users: list of the similar users
    :return: predicted rating
    """
    if has_users:
        similar_user_values = []
        for i, (key, value) in enumerate(similar_users.items()):
            if i > 5:
                break
            similar_user_values.append(utility_matrix[query][key])
        return sum(similar_user_values) / len(similar_user_values)
    else:
        return 0.0


def compute_prediction_on_query(utility_matrix: DataFrame, user: str, has_queries: bool, similar_query: Dict) -> float:
    """
    Predict the rating based on similar query, if there are no similar queries return 0.0
    :param utility_matrix:
    :param user:
    :param has_queries: if there are similar queries
    :param similar_query: dict with similar queries
    :return: predicted rating
    """
    if has_queries:
        partial_sum_num = 0.0
        partial_sum_den = 0.0
        for i, (q_id, score) in enumerate(sorted(similar_query.items(), key=lambda x: x[1], reverse=True)):
            if i > 5:
                break
            partial_sum_den += float(score)
            rating = utility_matrix[q_id][user]
            partial_sum_num += float(rating) * float(score)
        return partial_sum_num / partial_sum_den
    else:
        return 0.0


def fill_utility_matrix(utility_matrix: DataFrame, user_list: Set, queries) -> DataFrame:
    """
    Fill the missing values in the utility matrix
    :param utility_matrix:
    :param user_list:
    :param queries:
    :return:
    """
    print("Retrieving the set of queries...")
    queries_set = get_queries_set(queries)
    print("Computing similarity between users...")
    user_similarity_matrix = getUserSimilarityMatrix(utility_matrix)
    print("Computing similarity between queries...")
    query_similarity_matrix = get_query_similarity_scores(queries)
    print("Finding queries reviewed by each user...")
    user_query_list = get_query_for_user(utility_matrix)

    print("Filling the utility matrix...")
    for user in tqdm(user_list):
        # getting the list of queries rated by the user
        query_for_user = user_query_list[user]
        # computing the queries that still needs to be rated
        query_to_rate = queries_set.difference(query_for_user)
        # getting the list of similar users
        similar_users = get_similar_users(user, user_similarity_matrix)

        for curr_query in query_to_rate:
            # check if the similar users have rated the query that is being rated
            for (key, val) in similar_users.items():
                if pd.isna(utility_matrix[curr_query][key]):
                    similar_users.pop(key)

            if len(similar_users) == 0:
                has_similar_user = False
            else:
                has_similar_user = True

            # get the similar query for the user and query in consideration
            similar_query = query_similarity_matrix[curr_query]
            similar_query = dict(filter(lambda elem:
                                        (elem[1] > 0.0) and (elem[0] in query_for_user),
                                        similar_query.items()))
            if len(similar_query) == 0:
                has_similar_query = False
            else:
                has_similar_query = True

            if not has_similar_user and not has_similar_query:
                pred_value = random.uniform(1, 101)
            else:
                # Compute prediction based on similar users
                user_value = compute_prediction_on_user(utility_matrix, curr_query, has_similar_user, similar_users)

                # Compute prediction based on similar query
                query_value = compute_prediction_on_query(utility_matrix, user, has_similar_query, similar_query)

                # Combine the two values
                if has_similar_user and has_similar_query:
                    pred_value = (user_value + query_value) / 2
                elif has_similar_user:
                    pred_value = user_value
                elif has_similar_query:
                    pred_value = query_value

            utility_matrix.at[user, curr_query] = round(pred_value)
    utility_matrix.reset_index(inplace=True)
    return utility_matrix


def main(queries_file: str, table_file: str, user_list_file: str, utility_matrix_file: str):
    """
    Complete the utility matrix using similarity scores
    :param queries_file:
    :param table_file:
    :param user_list_file:
    :param utility_matrix_file:
    :return: Complete utility matrix
    """
    table = read_table(table_file)

    queries = read_queries(queries_file)
    queries = build_queries_on_table(table, queries)

    user_list = read_user_list(user_list_file)

    utility_matrix = read_utility_matrix(utility_matrix_file)

    filled_utility_matrix = fill_utility_matrix(utility_matrix, user_list, queries)

    filled_utility_matrix.to_csv(os.path.join(UTILITY_MATRICES_DIR, f"filled_{utility_matrix_file}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for filling the utility matrix")
    parser.add_argument("--table", type=str, default='movies.csv', help="Table file")
    parser.add_argument("--queries", type=str, default="movies.csv", help="Queries list file")
    parser.add_argument("--user-list", type=str, default="movies_user_list", help="User list file")
    parser.add_argument("--utility-matrix", type=str, default="movies_utility_matrix.csv", help="Utility matrix file")
    parser.add_argument("--domain", type=str, default="people",
                        help="Domain for the dataset (Available domains: movies, music, people)")
    args = parser.parse_args()

    if args.domain == "":
        queries_filename = args.queries
        table_filename = args.table
        user_list_filename = args.user_list
        utility_matrix_filename = args.utility_matrix
    else:
        domain = args.domain
        queries_filename = f"{domain}.csv"
        table_filename = f"{domain}.csv"
        user_list_filename = f"{domain}_user_list"
        utility_matrix_filename = f"{domain}_utility_matrix.csv"

    main(queries_filename, table_filename, user_list_filename, utility_matrix_filename)
