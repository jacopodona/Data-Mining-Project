import csv
import os
from typing import List

import numpy as np
import argparse


ASSETS_DIR = "assets"
UTILITY_MATRICES_DIR = "utility_matrices"
TABLE_DIR = "tables"
QUERY_DIR = "queries"
USER_DIR = "users"
THRESHOLD = 0.33


def get_utility_matrix(users: List, queries: int, output_file_path: str):
    for user in users:
        row = [user]
        for _ in range(queries):
            if np.random.uniform() >= THRESHOLD:
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
    query_ids_len = len(queries_ids)
    with open(os.path.join(UTILITY_MATRICES_DIR, output_file_path), 'w') as fp:
        fp.write(",".join(queries_ids))
        fp.write("\n")

    del queries, queries_ids

    return query_ids_len


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
    args = parser.parse_args()

    queries_file = args.queries
    user_file = args.users

    output_path = queries_file[:-4]+"_utility_matrix"+queries_file[-4:]
    queries_num = read_queries(queries_file, output_path)
    users_list = read_users(user_file)

    get_utility_matrix(users_list, queries_num, output_path)
