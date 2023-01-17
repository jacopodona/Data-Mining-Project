import os
import csv
import pandas as pd
import numpy as np

def read_queries(file_path):
    queries = []
    with open(file_path, 'r') as fp:
        csv_reader = csv.reader(fp)
        for row in csv_reader:
            if row:
                queries.append(row)
    queries = [query[0].split(",") for query in queries]
    queries_ids = [query[0] for query in queries]
    query_ids_len = len(queries_ids)

    return queries

def get_evaluation_split(df):
    header = df.columns.values.tolist()
    table = df.to_numpy()

