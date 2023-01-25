from typing import List, Dict

from numpy import isnan
from pandas import DataFrame, Series
from polars import LazyFrame
from tqdm import tqdm

ASSETS_DIR = "../DatasetGeneration/assets"
UTILITY_MATRICES_DIR = "../DatasetGeneration/utility_matrices"
TABLE_DIR = "../DatasetGeneration/tables"
QUERY_DIR = "../DatasetGeneration/queries"
USER_DIR = "../DatasetGeneration/users"


def get_similar_users(user: str, user_similarity_matrix: DataFrame) -> Series:
    user_list = user_similarity_matrix[user].sort_values(ascending=False)
    user_list.pop(user)
    return user_list


def get_query_for_user(utility_matrix: DataFrame) -> Dict:
    query_for_user_list = dict()
    for row in tqdm(utility_matrix.itertuples()):
        query_for_user_list[row.Index] = set()
        for key in row._fields:
            if key == "Index":
                continue
            else:
                val = getattr(row, key)
                if val is not None:
                    if not isnan(float(val)):
                        query_for_user_list[row.Index].add(key)
    return query_for_user_list


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

    return intersect/union


def build_matrix(raw_matrix: List[Dict]) -> Dict:
    """
    Build a matrix with a dict for direct access to the scores
    :param raw_matrix: list of comparisons scores
    :return: similarity matrix as a dict
    """
    clean_matrix = dict()
    for row in raw_matrix:
        id_1, id_2, score = row.values()
        if id_1 in clean_matrix.keys():
            clean_matrix[id_1][id_2] = score
        else:
            clean_matrix[id_1] = {id_2: score}

        if id_2 in clean_matrix.keys():
            clean_matrix[id_2][id_1] = score
        else:
            clean_matrix[id_2] = {id_1: score}

    return clean_matrix


def get_query_similarity_scores(queries: List[Dict]) -> Dict:
    """
    Get the similarity matrix of all the queries
    :param queries: list of all queries
    :return: similarity matrix
    """
    score_matrix = []
    for i, query1 in enumerate(tqdm(queries)):
        if i <= len(queries)-2:
            for query2 in queries[i+1:]:
                score = get_query_similarity_score(query1["condition"], query2["condition"])
                score_matrix.append({"id_1": query1["id"],
                                     "id_2": query2["id"],
                                     "score": score})
    return build_matrix(score_matrix)
