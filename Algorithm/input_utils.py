import csv
import os
from typing import List, Dict

import pandas as pd
import polars as pl
from pandas import DataFrame as pdDataFrame
from polars import DataFrame

ASSETS_DIR = "../DatasetGeneration/assets"
UTILITY_MATRICES_DIR = "../DatasetGeneration/utility_matrices"
TABLE_DIR = "../DatasetGeneration/tables"
QUERY_DIR = "../DatasetGeneration/queries"
USER_DIR = "../DatasetGeneration/users"


def read_user_list(user_list_file: str) -> set:
    """
    Load the list of user list
    :param user_list_file: filename
    :return: set of users
    """
    with open(os.path.join(USER_DIR, user_list_file), 'r') as fp:
        user_list = fp.readlines()

    user_set = set()
    for user in user_list:
        user_set.add(user.strip())
    return user_set


def read_table(table_file: str) -> DataFrame:
    """
    Load the table
    :param table_file: Filename of the table
    :return:
    """
    return pl.read_csv(os.path.join(TABLE_DIR, table_file),
                       has_header=True,
                       sep=",",
                       encoding="utf-8")


def read_utility_matrix(utility_matrix_file: str) -> pdDataFrame:
    """
    Load the utility matrix
    :param utility_matrix_file: Filename of the utility matrix
    :return:
    """

    return pd.read_csv(os.path.join(UTILITY_MATRICES_DIR, utility_matrix_file),
                       index_col=0)


def parse_queries(raw_queries: List[str]) -> List[Dict]:
    """
    Parse the query and split the condition field accordingly
    :param raw_queries: list of raw query data
    :return: list of the parsed query
    """
    query_parsed = []
    for query in raw_queries:
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


def get_queries_set(queries: List[Dict]) -> set:
    """
    Return the set of all the queries
    :param queries: list of all the queries
    :return: set of the queries
    """
    query_set = set()
    for query in queries:
        query_set.add(query["id"])
    return query_set
