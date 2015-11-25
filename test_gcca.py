'''
Driver for performing generalized CCA on speech data

'''

from gcca import *
from sklearn import neighbors

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
    
    number_of_views = len(data_locations)

    file_blocks = [1, 2, 3] # i.e. Use these blocks for training
    
    # Pre-process training data to have equal number of observations (n) 
    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            file_blocks, True)
    training_data_per_view, training_labels_per_view = data_pre_processor.process()
    
    # Perform GCCA on processed data
    gcca_model = GeneralizedCCA(training_data_per_view, 10)
    G = gcca_model.solve()
    
    # List for holding U_j for each view
    proj_matrix_per_view = list()
    
    training_data = np.ndarray(shape=(0, np.shape(G)[1]), dtype=np.float)
    training_labels = np.array([])
    
    # Compute U_j (matrix for projecting data into lower dimensional subspace)
    for i in range(number_of_views):
        U = np.linalg.pinv(training_data_per_view[i].transpose()) * np.mat(G)
        proj_matrix_per_view.append(U)
        training_data = np.vstack((training_data, np.mat(training_data_per_view[i].transpose()) * np.mat(U)))
        
        labels = training_labels_per_view[i]
        for label in labels:
            training_labels = np.hstack((training_labels, label[0]))
    
    # Fit k-NN model
    knn_model = neighbors.KNeighborsClassifier(15, weights='uniform')
    knn_model.fit(training_data, training_labels)
    
    # Start tuning
    file_blocks = [4] # i.e. Use these blocks for tuning
    
    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            file_blocks, False)
    tuning_data_per_view, tuning_labels_per_view = data_pre_processor.process()
    
    num_of_queries = 0.0
    num_of_matches = 0.0
    
    for i in range(number_of_views):
        predicted_labels = knn_model.predict(np.mat(tuning_data_per_view[i].transpose()) * np.mat(proj_matrix_per_view[i]))
        actual_labels = tuning_labels_per_view[i]
        
        for j in range(len(predicted_labels)):
            if predicted_labels[j] == actual_labels[j][0]:
                num_of_matches = num_of_matches + 1
            num_of_queries = num_of_queries + 1
    
    print num_of_matches / num_of_queries
