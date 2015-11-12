'''
GeneralizedCCA.py

Class for performing approximate regularized GCCA

'''

import numpy as np
import scipy.io as sio

class GeneralizedCCA:
    
    def __init__(self, data_locations, m_rank=0):
        '''
        Constructor for GeneralizedCCA.

        Args:
            data_locations (list<str>): Directories for each view's data
            m_rank (int): How many principal components to keep. A value of 0
                indicates that it should be full-rank. (Default 0)
        '''
        
        self.data_locations = data_locations
        self.m_rank = m_rank
    
    def solve(self):
        '''
        Solves MAX-VAR GCCA optimization problem and returns the matrix G

        Returns:
            numpy.ndarray, the matrix 'G' that solves GCCA optimization problem
        '''
        
        # Sequentially load data to scale up to large data sets
        
        for data_location in self.data_locations:
            # Load data for j-th view (X_j)
            mat_contents = sio.loadmat(data_location)
            
            # Perform rank-m SVD of X_j which yields X_j = A_j*S_j*B_j^T
            
            # Compute and store A_J*T_J where T_j*T_j^T = S_j^T(r_jI+S_jS_j^T)^(-1)S_j 
             
            pass
        
        # Create an N by mJ matrix 'M^tilde' which is given by [A_1*T_1 ... A_J*T_J] 
            
        # Perform SVD on M^tilde using incremental PCA (Brand, 2002) which yields G*S*V^T
        
        # Finally, return matrix G which has been compute from above