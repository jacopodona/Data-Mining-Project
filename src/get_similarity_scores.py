from typing import List, Dict

import numpy as np
import polars as pl
from numpy import isnan
from pandas import DataFrame
from polars import LazyFrame
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

ASSETS_DIR = "../data/assets"
UTILITY_MATRICES_DIR = "../data/utility_matrices"
TABLE_DIR = "../data/tables"
QUERY_DIR = "../data/queries"
USER_DIR = "../data/users"


def get_query_for_user(utility_matrix: DataFrame) -> Dict:
    query_for_user_list = dict()
    for row in tqdm(utility_matrix.itertuples()):
        query_for_user_list[row.Index] = set()
        # noinspection PyProtectedMember
        for key in row._fields:
            if key == "Index":
                continue
            else:
                val = getattr(row, key)
                if val is not None:
                    if not isnan(float(val)):
                        query_for_user_list[row.Index].add(key)
    return query_for_user_list


def get_similar_users(user: str, user_similarity_matrix: LazyFrame) -> LazyFrame:

    user_list = user_similarity_matrix \
        .filter(pl.col("index") != user) \
        .select(["index", user]) \
        .sort(user, reverse=True)
    return user_list


def get_user_similarity_matrix(utility_matrix: DataFrame) -> LazyFrame:
    """
    Build the user similarity matrix using the cosine similarity
    :param utility_matrix: raw utility matrix
    :return: user similarity matrix
    """
    ids = utility_matrix.index.tolist()
    table = utility_matrix.to_numpy()
    # Get a "normalized" utility matrix, where every item rating equals to the difference between the actual rating
    # in the 1-100 range minus the user average grade
    row_avg = np.nanmean(table, axis=1)  # Ignores NaN when computing mean
    normalized_matrix = table - row_avg[:, np.newaxis]
    # Replaces NaN values with 0s
    normalized_matrix = np.nan_to_num(normalized_matrix)
    matrix = csr_matrix(normalized_matrix)  # pandas DF into scipy sparse matrix

    # Compute user similarity matrix using cosine similarity
    similarities = cosine_similarity(matrix)
    # similarities = pd.DataFrame(similarities, index=ids, columns=ids)
    similarities = pl.DataFrame(similarities, columns=ids)
    similarities = similarities.with_columns([
        pl.Series(name="index", values=ids),
        pl.all()
    ]).lazy()
    return similarities


def get_query_similarity_score(query1: LazyFrame, query2: LazyFrame) -> float:
    """
    Use IoU with query results to compute similarity score between queries
    :param query1: first query
    :param query2: second query
    :return: similarity score between the specified queries
    """
    results1 = query1.collect()
    results2 = query2.collect()

    intersect = results1.join(results2, on="id", how="inner").height
    union = results1.join(results2, on="id", how="outer").height

    return intersect / union


def get_query_similarity_scores(queries: List[Dict]) -> Dict:
    """
    Get the similarity matrix of all the queries
    :param queries: list of all queries
    :return: similarity matrix
    """
    score_matrix = dict()

    for i, query1 in enumerate(tqdm(queries)):
        if i <= len(queries) - 2:
            for query2 in queries:
                score = get_query_similarity_score(query1["condition"], query2["condition"])

                if query1["id"] in score_matrix.keys():
                    score_matrix[query1["id"]][query2["id"]] = score
                else:
                    score_matrix[query1["id"]] = {query2["id"]: score}

                if query2["id"] in score_matrix.keys():
                    score_matrix[query2["id"]][query1["id"]] = score
                else:
                    score_matrix[query2["id"]] = {query1["id"]: score}

    return score_matrix
