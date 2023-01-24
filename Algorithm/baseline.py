import math
import random

import numpy as np
import utils
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from tqdm import tqdm

def getUserSimilarityMatrix(utility_matrix):
    table=utility_matrix.to_numpy()

    #Get a "normalized" utility matrix, where every item rating equals to the difference between the actual rating in the 1-100 range minus the user average grade
    row_avg = np.nanmean(table,axis=1)#Ignores NaN when computing mean
    normalized_matrix = table - row_avg[:, np.newaxis]

    #Replaces NaN values with 0s
    normalized_matrix=np.nan_to_num(normalized_matrix)
    matrix = scipy.sparse.csr_matrix(normalized_matrix)  # pandas DF into scipy sparse matrix

    #Compute user similarity matrix using cosine similarity
    similarities = cosine_similarity(matrix)
    return similarities

def top_K_highest_values(row,k):
    # get the indices and values of the top five highest values in the array
    top_five_indices = row.argsort()[-k:][::-1]
    top_five_values = row[top_five_indices]
    return top_five_indices,top_five_values

def fillUtilityMatrix(utility_matrix, similarity_matrix, topK):
    #Using a Knn style approach for filling the utility matrix behaviour
    header=utility_matrix.columns.values.tolist()
    original_matrix = utility_matrix.to_numpy()
    filled_matrix=utility_matrix.to_numpy()
    for i in tqdm(range(0, len(original_matrix))):
        row=original_matrix[i]
        for j in range(len(row)):
            rating=row[j]
            if(math.isnan(rating)): #Update only empty entries in utility matrix
                indices, weights = top_K_highest_values(similarity_matrix[0], k=topK + 1) #Since the maximum similarity of an user is with himself, we get k+1 top values and discard the maximum
                indices = indices[1:]#TO FIX, non usare i valori prediction, manda solo la row della matrix originale con i voti reali degli utenti
                weights = weights[1:]
                values=original_matrix[indices, j]
                prediction=0
                weights_sum=0
                for h in range(0,len(values)):
                    v=values[h]
                    if(not math.isnan(v)):
                        prediction+=v*weights[h]
                        weights_sum+=weights[h]
                try:
                    prediction=int(prediction/weights_sum)
                except ZeroDivisionError:
                    prediction=random.randint(1,100)
                row[j]=prediction
        filled_matrix[i]=row

    return filled_matrix



if __name__ == '__main__':
    print('Preparing dataset...')
    table = '../DatasetGeneration/tables/people.csv'
    queries = utils.read_queries('../DatasetGeneration/queries/people.csv')
    matrix='../DatasetGeneration/sparse_utility_matrices/people_utility_matrix.csv'

    items_table=pd.read_csv(table)
    utility_matrix=pd.read_csv(matrix)

    user_similarity=getUserSimilarityMatrix(utility_matrix)
    utility = utility_matrix.to_numpy()

    _, val_full_df = utils.get_train_val_split(utility_matrix, 0.8)
    val_masked_df = utils.mask_val_split(val_full_df, 0.7)

    train_utility_matrix=utils.replaceMatrixSection(utility_matrix,val_masked_df)

    filled_matrix=fillUtilityMatrix(utility_matrix,user_similarity,topK=5)

    full=val_full_df.to_numpy()
    masked=val_masked_df.to_numpy()
