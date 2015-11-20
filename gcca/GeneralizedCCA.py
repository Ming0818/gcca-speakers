'''
GeneralizedCCA.py

Class for performing approximate regularized GCCA

'''

import numpy as np
import scipy.io as sio

class GeneralizedCCA:
    
    def __init__(self, training_data_per_view, m_rank=0):
        '''
        Constructor for GeneralizedCCA.

        Args:
            training_data_per_view (list<ndarray>): Training data for each view
            m_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''
        
        self.training_data_per_view = training_data_per_view
        self.m_rank = m_rank
    
    def solve(self):
        '''
        Solves MAX-VAR GCCA optimization problem and returns the matrix G

        Returns:
            numpy.ndarray, the matrix 'G' that solves GCCA optimization problem
        '''
        
        reg = 0.00000001 # regularization parameter
        
        M = [] # matrix corresponding to M^tilde
        
        # Sequentially load data to scale up to large data sets
        
        for i in range(len(self.training_data_per_view)):
            X = self.training_data_per_view[i].transpose()
            
            # Perform rank-m SVD of X_j which yields X_j = A_j*S_j*B_j^T
            A,S,B = np.linalg.svd(X, full_matrices=False)
            
            S = np.diag(S)

            if self.m_rank != 0:
                A = A[:,0:self.m_rank]
            
            if self.m_rank != 0:
                S = S[:,0:self.m_rank]
                
            N = np.shape(A)[0]
            m = np.shape(S)[0]

            # Compute and store A_J*T_J where T_j*T_j^T = S_j^T(r_jI+S_jS_j^T)^(-1)S_j 
            T = np.sqrt(np.mat(S.transpose()) * np.linalg.inv(reg * np.identity(m) + np.mat(S) * np.mat(S.transpose())) * np.mat(S))
            
            # Create an N by mJ matrix 'M^tilde' which is given by [A_1*T_1 ... A_J*T_J] 
            if i == 0:
                M = np.array([], dtype=np.double).reshape(N,0)
            
            # Append to existing M^tilde (TODO: may need to use incremental update here, instead of storing full M)
            M = np.hstack((M, np.mat(A) * np.mat(T)))
        
        # Perform SVD on M^tilde which yields G*S*V^T (TODO: use incremental PCA (Brand, 2002))
        G,S,V = np.linalg.svd(M, full_matrices=False)
        
        # Finally, return matrix G which has been compute from above
        return G