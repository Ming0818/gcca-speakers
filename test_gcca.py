'''
Driver for performing generalized CCA on speech data

'''

from gcca import *
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import os

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

if __name__ == '__main__':
    # input parameters
    fold_number = 4
    num_of_neighbors = 4
    num_of_dimensions = 20
    data_directory = 'data/speech/'
    
    # Configure file locations
    data_locations = list()
    file_idx_location = None

    for file in os.listdir(data_directory):
        if file.startswith('JW') and file.endswith('.mat'):
            data_locations.append(os.path.join(data_directory, file))
        if file.endswith('fileidx.mat'):
            file_idx_location = os.path.join(data_directory, file)

    data_locations.sort()
    
    number_of_views = len(data_locations)
    number_of_folds = 5
    
    training_blocks = [0, 0, 0]
    tuning_blocks = [0]
    testing_blocks = [0]
    
    # Configure blocks depending on fold
    for i in range(number_of_folds):
        block = fold_number - 1 + i
        if block >= number_of_folds:
            block = block % number_of_folds
        if i <= 2:
            training_blocks[i] = block + 1
        elif i == 3:
            tuning_blocks[0] = block + 1
        else:
            testing_blocks[0] = block + 1
            
    # Pre-process training data to have equal number of observations (n) 
    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            training_blocks, True)
    training_data_per_view, training_labels_per_view = data_pre_processor.process()
    
    # Perform GCCA on processed data
    gcca_model = GeneralizedCCA(training_data_per_view, num_of_dimensions)
    G = gcca_model.solve()
    
    # List for holding U_j for each view
    proj_matrix_per_view = list()
    
    training_data = np.ndarray(shape=(0, np.shape(G)[1]), dtype=np.float)
    training_labels = np.array([], dtype=np.int)
    
    cmap = get_cmap(44)
    colors = []
    
    # Compute U_j (matrix for projecting data into lower dimensional subspace)
    for i in range(number_of_views):
        U = np.linalg.pinv(training_data_per_view[i].transpose()) * np.mat(G)
        
        projected_data = np.mat(training_data_per_view[i].transpose()) * np.mat(U)
        training_data = np.vstack((training_data, projected_data))
        
        proj_matrix_per_view.append(U)
        
        labels = training_labels_per_view[i]
        for label in labels:
            training_labels = np.hstack((training_labels, int(label)))
            colors.append(cmap(int(label)))
    
    plt.scatter(training_data[:,1], training_data[:,2], color=colors)
    plt.show()
    
    # Fit k-NN model
    knn_model = neighbors.KNeighborsClassifier(num_of_neighbors, weights='distance')
    knn_model.fit(training_data, training_labels)
    
    # Start tuning
    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            tuning_blocks, False)
    tuning_data_per_view, tuning_labels_per_view = data_pre_processor.process()
    
    num_of_queries = 0
    num_of_matches = 0
    
    for i in range(number_of_views):
        projected_data = np.mat(tuning_data_per_view[i].transpose()) * np.mat(proj_matrix_per_view[i])
        predicted_labels = knn_model.predict(projected_data)
        actual_labels = tuning_labels_per_view[i]
        
        for j in range(len(predicted_labels)):
            if int(predicted_labels[j]) == int(actual_labels[j]):
                num_of_matches = num_of_matches + 1
            num_of_queries = num_of_queries + 1
    
    print float(num_of_matches) / float(num_of_queries)
