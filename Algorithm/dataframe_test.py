
import random

import dask.dataframe as dd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import time
import utils


def try_pandas(table_path,query):
    print('--- PANDAS ---')
    start_time = time.time()
    df = pd.read_csv(table_path)
    print('Query=',query)
    print("Loading time: %s seconds " % round(time.time() - start_time,3))

    #query.pop(0) #Remove query id
    start_time = time.time()
    for i in range(1,len(query)):
        condition=query[i]
        attribute=condition.split('=')[0].strip()
        value=condition.split('=')[1]
        if(value.isnumeric()):
            value=int(value)
        df=df[(df[''+attribute]==value)]
    print('Number of results',len(df),'execution time=',round(time.time() - start_time,3))

def try_dask(table_path,query):
    #API: https://docs.dask.org/en/latest/dataframe.html
    print('--- DASK ---')
    start_time = time.time()
    df = dd.read_csv(table_path)
    df.head()
    print('Query=', query)
    print("Loading time: %s seconds " % round(time.time() - start_time, 3))

    # query.pop(0) #Remove query id
    start_time = time.time()
    for i in range(1, len(query)):
        condition = query[i]
        attribute = condition.split('=')[0].strip()
        value = condition.split('=')[1]
        if (value.isnumeric()):
            value = int(value)
        df = df[(df['' + attribute] == value)]
    print('Number of results', len(df), 'execution time=', round(time.time() - start_time, 3))

def try_spark(table_path,query):
    #API: https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html
    print('--- PYSPARK ---')
    start_time = time.time()
    spark = SparkSession \
        .builder \
        .appName("prova") \
        .getOrCreate()
    df = spark.read.csv(table_path)
    print('Query=', query)
    print("Loading time: %s seconds " % round(time.time() - start_time, 3))
    start_time = time.time()
    for i in range(1, len(query)):
        condition = query[i]
        attribute = condition.split('=')[0].strip()
        value = condition.split('=')[1]
        if (value.isnumeric()):
            value = int(value)
        #df=df.query(''+attribute+' == '+ 'value')
        #df = df.filter(col(attribute)==value)#df[(df['' + attribute] == value)]
    print('Number of results', df.count(), 'execution time=', round(time.time() - start_time, 3))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    table='../DatasetGeneration/tables/people.csv'
    queries=utils.read_queries('../DatasetGeneration/queries/people.csv')

    for i in range(0,5):
        print('Iteration:',i)
        query=random.choice(queries)
        try_pandas(table,query)
        try_dask(table, query)
        #try_spark(table,query)
        print('='*85)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
