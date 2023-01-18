import argparse
import csv
import os
from pprint import pprint
from typing import List, Dict

import polars as pl
from polars import DataFrame, LazyFrame
from tqdm import tqdm

ASSETS_DIR = "assets"
UTILITY_MATRICES_DIR = "utility_matrices"
TABLE_DIR = "tables"
QUERY_DIR = "queries"
USER_DIR = "users"
SIMILARITY_MATRICES_DIR = "similarity_matrix"


def get_similarity_score(query1: LazyFrame, query2: LazyFrame) -> float:
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


def get_similarity_scores(queries: List[Dict]) -> ():
    """
    Get the similarity matrix of all the queries
    :param queries: list of all queries
    :return:
    """
    score_matrix = []
    for i, query1 in enumerate(tqdm(queries)):
        if i <= len(queries)-2:
            for query2 in queries[i+1:]:
                score = get_similarity_score(query1["condition"], query2["condition"])
                score_matrix.append({"id_1": query1["id"],
                                     "id_2": query2["id"],
                                     "score": score})

    test_results = [res for res in score_matrix if (res["score"] > 0.0)]
    pprint(test_results)


def parse_queries(queries: List[str]) -> List[Dict]:
    """
    Parse the query and split the condition field accordingly
    :param queries: list of raw query data
    :return: list of the parsed query
    """
    query_parsed = []
    for query in queries:
        tmp = dict()
        tmp["id"] = query.split(",")[0]
        tmp["conditions"] = dict()
        for cond in query.split(",")[1:]:
            tmp["conditions"][cond.split("=")[0].strip()] = cond.split("=")[1]
        query_parsed.append(tmp)
    return query_parsed


def read_queries(queries_file: str) -> List[Dict]:
    """
    Read and parse the queries from the specified file
    :param queries_file: file name with the queries' information
    :return: list of parsed queries
    """
    queries = []
    with open(os.path.join(QUERY_DIR, queries_file), 'r') as fp:
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            if row:
                queries.extend(row)

    queries = parse_queries(queries)

    return queries


def build_queries_on_table(df: DataFrame, queries: List[Dict]) -> List[Dict]:
    """
    Build a list of dictionaries that associated with every
    query id the matching query expression on the Dataframe
    :param df: Datable with table's information
    :param queries: List of query to do on the table
    :return: List of the query id with the matching expression
    """
    df = df.lazy()
    queries_with_expressions = []
    for query in queries:
        tmp = dict()
        tmp["id"] = query["id"]
        expression = df
        for key, value in query["conditions"].items():
            if value.isnumeric():
                expression = expression.filter(pl.col(key) == int(value))
            else:
                expression = expression.filter(pl.col(key) == str(value))

        tmp["condition"] = expression
        queries_with_expressions.append(tmp)

    return queries_with_expressions


def get_similarity_matrix(table_file: str, queries_file: str) -> ():
    """
    Build the similarity matrix
    :param table_file: file containing the table with information
    :param queries_file: file containing the list of queries
    :return:
    """
    df = pl.read_csv(os.path.join(TABLE_DIR, table_file), has_header=True, sep=",", encoding="utf-8")

    queries = read_queries(queries_file)

    queries = build_queries_on_table(df, queries)

    get_similarity_scores(queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for similarity matrix generation")
    parser.add_argument("--table", type=str, default='people.csv',
                        help="Table file")
    parser.add_argument("--queries", type=str, default="people.csv", help="Queries list file")
    args = parser.parse_args()

    queries_filename = args.queries
    table_filename = args.table

    get_similarity_matrix(table_filename, queries_filename)
