'''
Driver for performing generalized CCA on speech data

'''

from gcca import *

import numpy as np
import os

if __name__ == '__main__':
    data_directory = 'data/speech/'

    data_locations = list()
    file_idx_location = None

    for file in os.listdir(data_directory):
        if file.startswith('JW') and file.endswith('.mat'):
            data_locations.append(os.path.join(data_directory, file))
        if file.endswith('fileidx.mat'):
            file_idx_location = os.path.join(data_directory, file)

    data_locations.sort()

    file_blocks = [1, 2, 3] # i.e. Use these blocks for training
    
    # Pre-process data to have equal number of observations (n) 
    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            file_blocks)
    training_data_per_view, training_labels_per_view = data_pre_processor.process()
    
    # Perform GCCA on processed data
    gcca_model = GeneralizedCCA(training_data_per_view, 10)
    G = gcca_model.solve()
    
    # List for holding U_j for each view
    proj_matrix_per_view = list()
    
    # Compute U_j (matrix for projecting data into lower dimensional subspace)
    for i in range(len(training_data_per_view)):
        training_data = training_data_per_view[i]
        U = np.linalg.pinv(training_data.transpose()) * np.mat(G)
        proj_matrix_per_view.append(U)
