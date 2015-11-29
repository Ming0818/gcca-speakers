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
import sys
from __builtin__ import str


# "Enum" to specify different face detection data sets
class ClassificationModel:
    K_Neighbors = 'K Nearest Neighbors'
    Kernel_SVM_RBF = 'Kernel SVM - RBF'

vowel_labels = [0, 1, 3, 10, 17, 24, 33]
num_of_dimensions = 0 # 0 for full number of dimensions
classification_model = ClassificationModel.Kernel_SVM_RBF
use_full_phones = True

def getColorMap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def getAccuracies(model, data_locations, file_idx_location, blocks, proj_matrix_per_view):
    number_of_views = len(data_locations)

    data_pre_processor = DataPreProcessor(data_locations, file_idx_location,
            blocks, False)
    data_per_view, labels_per_view = data_pre_processor.process()

    accuracies = []

    num_of_queries = 0
    num_of_matches = 0

    for i in range(number_of_views):
        num_of_queries_per_view = 0
        num_of_matches_per_view = 0

        projected_data = np.mat(data_per_view[i].transpose()) * np.mat(proj_matrix_per_view[i])
        actual_labels = labels_per_view[i]
        
        query_data = np.ndarray(shape=(0, np.shape(projected_data)[1]), dtype=np.float)
        query_labels = np.array([], dtype=np.int)
        
        if use_full_phones:
            query_data = np.vstack((query_data, projected_data))
            
            for j in range(len(actual_labels)):
                query_labels = np.hstack((query_labels, int(actual_labels[j])))
        else:
            for j in range(len(actual_labels)):
                if (actual_labels[j] in vowel_labels):
                    query_data = np.vstack((query_data, projected_data[j,:]))
                    query_labels = np.hstack((query_labels, int(actual_labels[j])))

        predicted_labels = model.predict(query_data)

        for j in range(len(predicted_labels)):
            if int(predicted_labels[j]) == int(test_labels[j]):
                num_of_matches = num_of_matches + 1
                num_of_matches_per_view = num_of_matches_per_view + 1
            num_of_queries = num_of_queries + 1
            num_of_queries_per_view = num_of_queries_per_view + 1

        accuracies.append(float(num_of_matches_per_view) / float(num_of_queries_per_view))

    accuracies.append(float(num_of_matches) / float(num_of_queries))

    return accuracies

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
        
        if use_full_phones:
            training_data = np.vstack((training_data, projected_data))
            
            for j in range(len(labels)):
                colors.append(cmap(int(labels[j])))
                training_labels = np.hstack((training_labels, int(labels[j])))
        else:
            for j in range(len(labels)):
                if (labels[j] in vowel_labels):
                    training_data = np.vstack((training_data, projected_data[j,:]))
                    training_labels = np.hstack((training_labels, int(labels[j])))
                    colors.append(cmap(vowel_labels.index(int(labels[j]))))

    #plot = plt.scatter(training_data[:,2], training_data[:,1], color=colors)
    #plt.show()

    # Start tuning/testing
    if classification_model == ClassificationModel.Kernel_SVM_RBF:
        max_accuracy = 0.0
        optimal_gamma = 0

        for i in [100, 200, 300, 400, 500]:
            model = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=i,C=1000)
            model.fit(training_data, training_labels)
            accuracies = getAccuracies(model, data_locations, file_idx_location, tuning_blocks, proj_matrix_per_view)
            if accuracies[len(accuracies) - 1] > max_accuracy:
                max_accuracy = accuracies[len(accuracies) - 1]
                optimal_gamma = i

        print '| Optimal gamma value: {}'.format(optimal_gamma)

        model = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=optimal_gamma,C=1000)
    else:
        max_accuracy = 0.0
        optimal_neighbors = 0
        for i in [4, 8, 12, 16]:
            model = neighbors.KNeighborsClassifier(i, weights='distance')
            model.fit(training_data, training_labels)
            accuracies = getAccuracies(model, data_locations, file_idx_location, tuning_blocks, proj_matrix_per_view)
            if accuracies[len(accuracies) - 1] > max_accuracy:
                max_accuracy = accuracies[len(accuracies) - 1]
                optimal_neighbors = i

        print '| Optimal number of neighbors: {}'.format(optimal_neighbors)

        model = neighbors.KNeighborsClassifier(optimal_neighbors, weights='distance')

    model.fit(training_data, training_labels)
    accuracies = getAccuracies(model, data_locations, file_idx_location, testing_blocks, proj_matrix_per_view)

    for i in range(len(accuracies)):
        if i < len(accuracies) - 1:
            print '| Accuracy for view {}: {:.3f}'.format(i + 1, accuracies[i])
        else:
            print '| Accuracy for whole data: {:.3f}'.format(accuracies[i])

    print '|'

if __name__ == '__main__':

    data_directory = 'data/speech/'

    # If provided a command-line argument, use that as the data location.
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]

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





