import time

import numpy as np

'''import warnings
warnings.filterwarnings("error")'''


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    start_time = time.time()
    Q = Q.T
    for step in range(0,steps):
        print(step)
        for i in range(0,len(R)):
            for j in range(0,len(R[i])):
                if R[i][j] > 0:
                #if not math.isnan(R[i][j]):
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(0,K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(0,len(R)):
            for j in range(0,len(R[i])):
                if R[i][j] > 0:
                #if not math.isnan(R[i][j]):
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(0,K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    print('Matrix factorization:',time.time()-start_time)
    return P, Q.T

def fillMatrix(utility_matrix):
    utility_matrix = utility_matrix.fillna(0)
    R = utility_matrix.to_numpy()


    N = len(R)
    M = len(R[0])
    K = 2 #Number of features/concepts

    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    start_time = time.time()
    nR = np.dot(nP, nQ.T)
    print('Matrix filling:', time.time() - start_time)
    return nR

'''if __name__ == '__main__':
    table = '../data/tables/people.csv'
    queries = utils.read_queries('../data/queries/people.csv')
    matrix='../data/utility_matrices/people_utility_matrix.csv'

    df=pd.read_csv(table)
    utility_matrix=pd.read_csv(matrix)
    #utility_matrix=utils.convertNaN(utility_matrix)
    #computeUserSimilarity(utility_matrix)
    train_df, val_full_df = utils.get_train_val_split(utility_matrix, 0.8)
    val_masked_df = utils.mask_val_split(val_full_df, 0.5)

    predicted=fillMatrix(utility_matrix)

    _,predicted_val=utils.get_train_val_split(predicted)

    print('Average error is: ',utils.evaluate(val_full_df,predicted_val))
    test_table = pd.DataFrame(np.ones(shape=np.shape(val_full_df.to_numpy())),
                              columns=val_masked_df.columns.values.tolist())
    print('Random error is: ', utils.evaluate(val_full_df, test_table))'''