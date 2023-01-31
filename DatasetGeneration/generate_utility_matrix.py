import argparse
import csv
import os
import random
from random import uniform, shuffle
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

ASSETS_DIR = "assets"
UTILITY_MATRICES_DIR = "utility_matrices"
TABLE_DIR = "tables"
QUERY_DIR = "queries"
USER_DIR = "users"


# THRESHOLD = 0.33

def get_query_similarity(query1, query2):
    """
    Computes similarity score between two queries given the conditions: same attribute in query ->+1 point, same attribute-value-->+5 point
    :param query1:
    :param query2:
    :return:
    """
    score = 0
    attributes1 = []
    value1 = []
    attributes2 = []
    value2 = []
    for i in range(1, len(query1)):
        condition = query1[i]
        attributes1.append(condition.split('=')[0].strip())
        value1.append(condition.split('=')[1])
    for i in range(1, len(query2)):
        condition = query2[i]
        attributes2.append(condition.split('=')[0].strip())
        value2.append(condition.split('=')[1])
    for i in range(len(attributes1)):
        field = attributes1[i]
        if field in attributes2:
            score += 1
            position = attributes2.index(attributes1[i])
            if (value1[i] == value2[position]):
                score += 5
    return score


def get_realistic_utility_matrix(users_list, queries, output_path):
    """
    Creates a semi-realistic utility matrix by giving similar score to similar queries sampling from normal distribution
    :param users_list:
    :param queries:
    :param output_path:
    :return:
    """
    # First give random rating to list of a small set of queries
    for i in tqdm(range(len(users_list))):
        user = users_list[i]
        num_rated_queries = abs(int(np.random.normal(loc=0.03, scale=0.02, size=1) * len(queries)))
        random_order_queries = queries
        shuffle(random_order_queries)
        rated_queries = random_order_queries[:num_rated_queries]
        rates = []
        for j in range(0, len(rated_queries)):
            rates.append(np.random.randint(0, 101))
        # Queries to rate using previous queries
        query_rate_results = abs(int(np.random.normal(loc=0.05, scale=0.02, size=1) * len(queries)))
        for j in range(0, query_rate_results):
            # Select a non rated query
            query_to_rate = random.choice(queries)
            while query_to_rate in rated_queries:
                query_to_rate = random.choice(queries)
            # Compute vote based on similarity with rated queries
            max_similarity = 0
            similar_query_index = -1
            for k in range(0, len(rated_queries)):
                pairwise_similiarity = get_query_similarity(query_to_rate, rated_queries[k])
                if (pairwise_similiarity > max_similarity):
                    max_similarity = pairwise_similiarity
                    similar_query_index = k
            if max_similarity == 0:  # No rated query is similar to selected query
                vote = np.random.randint(0, 101)
            else:
                standard_deviation = 1 / max_similarity  # The higher the score, the higher the similarity, the smaller the standard deviation from which the rating is drawn
                vote = rates[similar_query_index] + int(np.random.normal(loc=0, scale=standard_deviation*2,
                                                                         size=1))  # New vote is the one of the most similar query + noise
            if vote < 0:  # Clip value between 0 and 100 range
                vote = 0
            if vote > 100:
                vote = 100
            rated_queries.append(query_to_rate)
            rates.append(vote)
        writeUserRow(user, queries, rated_queries, rates, output_path)


def writeUserRow(user, total_queries, user_rated_queries, rates, output_path):
    row = [user]
    for i in range(0, len(total_queries)):
        q = total_queries[i]
        has_been_rated = False
        for j in range(0, len(user_rated_queries)):
            r = user_rated_queries[j]
            if q[0] == r[0]:  # If they have same id, the user has rated the query, append the vote
                row.append(str(rates[j]))
                has_been_rated = True
        # If they id has not been found the query has not been rated by the user, append empty rating
        if not has_been_rated:
            row.append("")
    with open(os.path.join(UTILITY_MATRICES_DIR, output_path), 'a') as fp:
        fp.write(",".join(row))
        fp.write("\n")


def get_utility_matrix(users: List, queries: int, output_file_path: str):
    for user in users:
        row = [user]
        #
        threshold = uniform(0.33, 0.5)
        for _ in range(queries):
            if np.random.uniform() >= threshold:
                row.append(str(np.random.randint(0, 101)))
            else:
                row.append("")
        with open(os.path.join(UTILITY_MATRICES_DIR, output_file_path), 'a') as fp:
            fp.write(",".join(row))
            fp.write("\n")


def read_queries(file_path: str, output_file_path: str) -> int:
    queries = []
    with open(os.path.join(QUERY_DIR, file_path), 'r') as fp:
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            if row:
                queries.append(row)

    queries = [query[0].split(",") for query in queries]
    queries_ids = [query[0] for query in queries]
    with open(os.path.join(UTILITY_MATRICES_DIR, output_file_path), 'w') as fp:
        fp.write(",".join(queries_ids))
        fp.write("\n")

    return queries


def read_users(file_path: str) -> List:
    with open(os.path.join(USER_DIR, file_path), 'r') as fp:
        user_list = fp.readlines()
    user_list = [user.strip() for user in user_list]
    return user_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for utility matrix generation")
    parser.add_argument("--users", type=str, default='user_list',
                        help="User list file")
    parser.add_argument("--queries", type=str, default="queries.csv", help="Queries list file")
    parser.add_argument("--domain", type=str, default="movies",
                        help="Domain for the dataset (Available domains: movies, music, people)")
    args = parser.parse_args()

    queries_file = args.queries
    user_file = args.users
    domain = args.domain
    table_path = os.path.join('tables', domain + '.csv')
    table_dataframe = pd.read_csv(table_path)
    output_path = queries_file[:-4] + "_utility_matrix" + queries_file[-4:]
    if not domain == "null":
        queries_file = f"{domain}.csv"
        user_file = f"{domain}_user_list"
        output_path = f"{domain}_utility_matrix.csv"
    # queries_num = read_queries(queries_file, output_path)
    users_list = read_users(user_file)
    queries = read_queries(queries_file, output_path)
    print('Generating utility matrix for users')
    # get_utility_matrix(users_list, queries_num, output_path)
    get_realistic_utility_matrix(users_list, queries, output_path)
