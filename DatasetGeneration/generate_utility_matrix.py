import csv
import os
import numpy as np


ASSETS_DIR = "assets"
THRESHOLD = 0.5


def get_utility_matrix(user_list, queries):
    matrix = [queries]
    for user in user_list:
        row = [user]
        for _ in queries:
            if np.random.uniform() >= THRESHOLD:
                row.append(str(np.random.randint(0, 101)))
            else:
                row.append("")
        matrix.append(row)

    with open(os.path.join(ASSETS_DIR, "utility_matrix.csv"), 'w') as fp:
        for row in matrix:
            fp.write(",".join(row))
            fp.write("\n")


if __name__ == "__main__":
    queries = []
    with open(os.path.join(ASSETS_DIR, "people-1.csv"), 'r') as fp:
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            if row != []:
                queries.append(row)

    queries = [query[0].split(",") for query in queries]
    queries_ids = [query[0] for query in queries]
    query_ids_len = len(queries_ids)

    with open(os.path.join(ASSETS_DIR, "user_list"), 'r') as fp:
        user_list = fp.readlines()
    user_list = [user.strip() for user in user_list]

    get_utility_matrix(user_list, queries_ids)
