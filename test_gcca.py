'''
test_gcca.py

Driver for performing generalized CCA on speech data.

'''

from gcca import *
import util
from util import ClassificationModel

from sklearn import neighbors
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

import os
import sys


vowel_labels = [0, 1, 3, 10, 17, 24, 33]
num_of_dimensions = 0 # 0 for full number of dimensions
classification_model = ClassificationModel.Kernel_SVM_RBF
use_full_phones = False


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
            if int(predicted_labels[j]) == int(query_labels[j]):
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

    ( training_blocks,
      tuning_blocks,
      testing_blocks ) = util.configure_blocks(fold_number)

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
        cmap = util.getColorMap(38)
    else:
        cmap = util.getColorMap(len(vowel_labels))

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

    data_directory = 'data/speech'

    # If provided a command-line argument, use that as the data location.
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]

    data_locations, file_idx_location = util.find_file_locations(data_directory)

    # Print info
    print ('\n'
           'Data Directory:       {}\n'
           'Classification Model: {}\n'
           'Number of Dimensions: {}\n'
           'Using Full Phones:    {}\n'
    ).format(data_directory, classification_model, num_of_dimensions,
            use_full_phones)

    for fold_number in [1, 2, 3, 4, 5]:
        runSingleFold(data_locations, file_idx_location, fold_number)


