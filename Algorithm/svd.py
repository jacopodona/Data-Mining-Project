import math
import random
import numpy as np
import utils
import pandas as pd
import scipy
from numpy.linalg import norm as np_norm
import scipy.sparse as ss
from scipy.sparse.linalg import norm as ss_norm
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import math, random
from tqdm import tqdm
from numpy.linalg import svd

def generate_matrix(rows, columns):
    return np.random.uniform(1, 100, size=(rows, columns))


def performSVD(utility_matrix):
    matrix=utility_matrix.to_numpy()
    '''start_time=time.time()
    u,s,vh=scipy.sparse.linalg.svds(matrix)
    print('SVD computation:', time.time() - start_time)'''
    random_numgen = np.random.default_rng()
    m, n = 200, 50
    #array_data = random_numgen.standard_normal((m, n))
    array_data=generate_matrix(m,n)
    for i in range(0,m):
        for j in range(0,n):
            if(random.randint(0,100)>60):
                array_data[i,j]=0
    #U, s, Vh = scipy.linalg.svd(array_data,check_finite=True)
    U, s, Vh = scipy.sparse.linalg.svds(array_data,k=10)

    print("left singular vectors of Unitary matrix", U.shape)
    print("singular values", s)
    print("right singular vectors of Unitary matrix", Vh.shape)
    predictions=np.matmul(U*s,Vh)
    pass

def SVT(M: pd.DataFrame,
        max_iter: int = 1500,
        delta: int = 2,
        tolerance: float = 0.001,
        increment: int = 5,
        s: int = 2):
    """
    Params:
        M: matrix to complite
        max_iter: maximum number of iterations
        delta: step-size
        tolerance: tolerance on the minimum improvement
        increment: how many new singular values to check if they fall below tau
    Returns:
        X, rmse: complited matrix, error list
    """
    M = ss.csr_matrix(M.fillna(0).values)  # pandas DF into scipy sparse matrix
    n, m = M.shape

    total_num_nonzero = len(M.nonzero()[0])
    idx = random.sample(range(total_num_nonzero), int(total_num_nonzero))
    Omega = (M.nonzero()[0][idx], M.nonzero()[1][idx])

    tau = 5 * math.sqrt(n * m)

    ######
    # SVT
    ######
    r = 0
    rmse = []
    data, indices = np.ravel(M[Omega]), Omega
    P_Omega_M = ss.csr_matrix((data, indices), shape=(n, m))
    k_0 = np.ceil(tau / (delta * ss_norm(P_Omega_M)))  # element-wise ceiling
    Y = k_0 * delta * P_Omega_M

    for _ in tqdm(range(max_iter), desc= "Iteratively filling the matrix", colour="green"):
        #s = r + 1
        while True:
            # print("Y: ", Y)                               #svd(ss.csc_matrix(Y).toarray(), s) PIETRO USA QUESTO
            U, S, V = svd(ss.csc_matrix(Y).toarray(),s)      #svds(Y, 20)  # sparsesvd(ss.csc_matrix(Y), s)
            s += increment
            try:
                if S[s - increment] <= tau: break
            except:
                break

        r = np.sum(S > tau)

        U = U[:, :r]
        S = S[:r] - tau
        V = V[:r, :]

        # print(U.shape)
        # print(type(U))
        # print(S.shape)
        # print(type(S))
        # print(V.shape)
        # print(type(V))
        #
        # print(U)
        # print(S)
        # print(V)
        X = (U * S).dot(V)
        # print(X.shape)
        # print(X)
        # break


        #print(X.shape)
        X_omega = ss.csr_matrix((X[Omega], Omega), shape=(n, m))

        if ss_norm(X_omega - P_Omega_M) / ss_norm(P_Omega_M) < tolerance: break

        diff = P_Omega_M - X_omega
        Y += delta * diff
        # print(Y.shape)
        rmse.append(np_norm(M[M.nonzero()] - X[M.nonzero()]) / np.sqrt(len(X[M.nonzero()])))
        X = X.clip(0, 1)

    return X, rmse


def fulltest(train_utility_matrix):
    for i in range(2,100,2):
        print('K=',i)
        X, rmse = SVT(train_utility_matrix, max_iter=1500,s=i)
        prediction = pd.DataFrame(X)
        _, val_prediction_split = utils.get_train_val_split(prediction, 0.8)
        result=utils.evaluate(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split)
        #print('Prediction error per item is:',utils.evaluate(gt_df=val_full_df, masked_df=val_masked_df, proposal_df=val_prediction_split))
        print('Prediction error per item is:',result,'RSME=',min(rmse))

        with open("results_svd_nosparse.txt", "a") as file1:
            # Writing data to a file
            string=''+str(i)+' Prediction error per item is: '+str(result)+' RSME='+str(min(rmse))+'\n'
            file1.write(string)


if __name__ == '__main__':
    table = '../DatasetGeneration/tables/people.csv'
    queries = utils.read_queries('../DatasetGeneration/queries/people.csv')
    matrix='../DatasetGeneration/utility_matrices/people_utility_matrix.csv'

    items_table=pd.read_csv(table)
    utility_matrix=pd.read_csv(matrix)

    #utility_matrix=utils.convertNaN(utility_matrix)
    #computeUserSimilarity(utility_matrix)
    _, val_full_df = utils.get_train_val_split(utility_matrix, 0.8)
    val_masked_df = utils.mask_val_split(val_full_df, 0.7)

    train_utility_matrix=utils.replaceMatrixSection(utility_matrix,val_masked_df)

    utility=utility_matrix.to_numpy()
    full=val_full_df.to_numpy()
    masked=val_masked_df.to_numpy()
    X, rmse = SVT(train_utility_matrix, max_iter=1500)
    print(pd.DataFrame(X).head(15))
    prediction=pd.DataFrame(X)
    _,val_prediction_split=utils.get_train_val_split(prediction, 0.8)
    print('Prediction error per item is:',utils.evaluate(gt_df=val_full_df,masked_df=val_masked_df,proposal_df=val_prediction_split))



    '''x_coordinate = range(len(rmse))
    plt.ylim(0, 0.5)
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.plot(x_coordinate, rmse, '-')
    plt.show()'''

