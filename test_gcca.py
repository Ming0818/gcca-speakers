'''
Driver for performing generalized CCA on speech data

'''

from gcca import *
from sklearn import neighbors
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import os
from __builtin__ import str


# "Enum" to specify different face detection data sets
class ClassificationModel:
    K_Neighbors = 'K Nearest Neighbors'
    Kernel_SVM_RBF = 'Kernel SVM - RBF'
    Kernel_SVM_Poly = 'Kernel SVM - Polynomial'

vowel_labels = [0, 1, 3, 10, 17, 24, 33]
num_of_dimensions = 0 # 0 for full number of dimensions
classification_model = ClassificationModel.K_Neighbors
use_full_phones = True

def getColorMap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def getAccuracy(model, data_locations, file_idx_location, blocks, proj_matrix_per_view):
    number_of_views = len(data_locations)
    
    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            blocks, False)
    data_per_view, labels_per_view = data_pre_processor.process()
    
    num_of_queries = 0
    num_of_matches = 0
    
    for i in range(number_of_views):
        projected_data = np.mat(data_per_view[i].transpose()) * np.mat(proj_matrix_per_view[i])
        
        query_data = np.ndarray(shape=(0, np.shape(projected_data)[1]), dtype=np.float)
        test_labels = np.array([], dtype=np.int)
        
        actual_labels = labels_per_view[i]
        
        for j in range(len(actual_labels)):
            if use_full_phones or (actual_labels[j] in vowel_labels):
                query_data = np.vstack((query_data, projected_data[j,:]))
                test_labels = np.hstack((test_labels, int(actual_labels[j])))
        
        predicted_labels = model.predict(query_data)
            
        for j in range(len(predicted_labels)):
            if int(predicted_labels[j]) == int(test_labels[j]):
                num_of_matches = num_of_matches + 1
            num_of_queries = num_of_queries + 1
    
    return float(num_of_matches) / float(num_of_queries)

def runSingleFold(data_locations, file_idx_location, fold_number):
    print '| ---- ---- Fold #{} ---- ----'.format(fold_number)
    
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
    
    if use_full_phones:
        cmap = getColorMap(38)
    else:
        cmap = getColorMap(len(vowel_labels))
        
    colors = []
    
    # Compute U_j (matrix for projecting data into lower dimensional subspace)
    for i in range(number_of_views):
        U = np.linalg.pinv(training_data_per_view[i].transpose()) * np.mat(G)
        
        projected_data = np.mat(training_data_per_view[i].transpose()) * np.mat(U)
        
        proj_matrix_per_view.append(U)
        
        labels = training_labels_per_view[i]
        for j in range(len(labels)):
            if use_full_phones or (labels[j] in vowel_labels):
                training_data = np.vstack((training_data, projected_data[j,:]))
                training_labels = np.hstack((training_labels, int(labels[j])))
                if use_full_phones:
                    colors.append(cmap(int(labels[j])))
                else:
                    colors.append(cmap(vowel_labels.index(int(labels[j]))))
    
    #plot = plt.scatter(training_data[:,2], training_data[:,1], color=colors)
    #plt.legend([plot, plot, plot, plot, plot, plot, plot],['AA', 'AE', 'AO', 'EH', 'IY', 'OW', 'UW'])
    #plt.show()
    
    # Start tuning/testing
    if classification_model == ClassificationModel.Kernel_SVM_RBF:
        model = svm.SVC(decision_function_shape='ovo',kernel='rbf')
        print getAccuracy(model, data_locations, file_idx_location, tuning_blocks, proj_matrix_per_view)
    elif classification_model == ClassificationModel.Kernel_SVM_Poly:
        model = svm.SVC(decision_function_shape='ovo',kernel='poly',degree=2,coef0=0)
        print getAccuracy(model, data_locations, file_idx_location, tuning_blocks, proj_matrix_per_view)
    else:
        max_accuracy = 0.0
        optimal_neighbors = 0
        for i in [4, 8, 12, 16]:
            model = neighbors.KNeighborsClassifier(i, weights='distance')
            model.fit(training_data, training_labels)
            accuracy = getAccuracy(model, data_locations, file_idx_location, tuning_blocks, proj_matrix_per_view)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                optimal_neighbors = i
        
        print '| Optimal number of neighbors: {}'.format(optimal_neighbors)
        
        model = neighbors.KNeighborsClassifier(optimal_neighbors, weights='distance')
    
    model.fit(training_data, training_labels)
    accuracy = getAccuracy(model, data_locations, file_idx_location, testing_blocks, proj_matrix_per_view)
    
    print '| Accuracy on test data: {:.3f}'.format(accuracy)
    print '|'

if __name__ == '__main__':
    data_directory = '../data/speech/'
    
    # Configure file locations
    data_locations = list()
    file_idx_location = None

    for file in os.listdir(data_directory):
        if file.startswith('JW') and file.endswith('.mat'):
            data_locations.append(os.path.join(data_directory, file))
        if file.endswith('fileidx.mat'):
            file_idx_location = os.path.join(data_directory, file)

    data_locations.sort()
    
    for fold_number in [1, 2, 3, 4, 5]:
        runSingleFold(data_locations, file_idx_location, fold_number)
    
    
                
    
    
