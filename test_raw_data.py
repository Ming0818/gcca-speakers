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

import scipy.io as sio

import os
import sys


vowel_labels = [0, 1, 3, 10, 17, 24, 33]
classification_model = ClassificationModel.Kernel_SVM_RBF
use_full_phones = False


def getAccuracy(model, data, labels):
    query_data = np.ndarray(shape=(0, np.shape(data)[0]), dtype=np.float)
    query_labels = np.array([], dtype=np.int)
        
    if use_full_phones:
        query_data = data.transpose()
        query_labels = labels
    else:
        for j in range(len(labels)):
            if (labels[j] in vowel_labels):
                query_data = np.vstack((query_data, data.transpose()[j,:]))
                query_labels = np.hstack((query_labels, int(labels[j])))

    predicted_labels = model.predict(query_data)
    
    num_of_matches = 0

    for j in range(len(predicted_labels)):
        if int(predicted_labels[j]) == int(query_labels[j]):
            num_of_matches = num_of_matches + 1

    return float(num_of_matches) / float(len(predicted_labels))

def runSingleFold(data_list, file_idx, fold_number):
    print '| ---- ---- Fold #{} ---- ----'.format(fold_number)

    number_of_views = len(data_list)

    ( training_blocks,
      tuning_blocks,
      testing_blocks ) = util.configure_blocks(fold_number)

    data_pre_processor = DataPreProcessor(data_list, file_idx,
            training_blocks, False)
    training_data_per_view, training_labels_per_view = data_pre_processor.process()
    
    data_pre_processor = DataPreProcessor(data_list, file_idx,
            tuning_blocks, False)
    tuning_data_per_view, tuning_labels_per_view = data_pre_processor.process()
    
    data_pre_processor = DataPreProcessor(data_list, file_idx,
            testing_blocks, False)
    testing_data_per_view, testing_labels_per_view = data_pre_processor.process()

    for i in range(number_of_views):
        training_data = np.ndarray(shape=(0, np.shape(training_data_per_view[i])[0]), dtype=np.float)
        training_labels = np.array([], dtype=np.int)
            
        if use_full_phones:
            training_data = training_data_per_view[i].transpose()
            training_labels = training_labels_per_view[i]
        else:
            for j in range(len(training_labels_per_view[i])):
                if (training_labels_per_view[i][j] in vowel_labels):
                    training_data = np.vstack((training_data, training_data_per_view[i].transpose()[j,:]))
                    training_labels = np.hstack((training_labels, int(training_labels_per_view[i][j])))
        
        param_msg = None
        
        # Start tuning/testing
        if classification_model == ClassificationModel.Kernel_SVM_RBF:
            max_accuracy = 0.0
            optimal_gamma = 0.0

            for j in [3e-08, 3.5e-08, 4e-08, 4.5e-08]:
                model = svm.SVC(decision_function_shape='ovo',kernel='rbf',gamma=j,C=2)
                model.fit(training_data, training_labels)
                accuracy = getAccuracy(model, tuning_data_per_view[i], tuning_labels_per_view[i])
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_gamma = j

            param_msg = 'Optimal gamma value: {}'.format(optimal_gamma)

            model = svm.SVC(decision_function_shape='ovo',kernel='rbf',gamma=optimal_gamma,C=2)
        else:
            max_accuracy = 0.0
            optimal_neighbors = 0
            
            for j in [28, 32, 36, 40]:
                model = neighbors.KNeighborsClassifier(j, weights='distance')
                model.fit(training_data, training_labels)
                accuracy = getAccuracy(model, tuning_data_per_view[i], tuning_labels_per_view[i])
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_neighbors = j

            param_msg = 'Optimal number of neighbors: {}'.format(optimal_neighbors)

            model = neighbors.KNeighborsClassifier(optimal_neighbors, weights='distance')

        model.fit(training_data, training_labels)
        accuracy = getAccuracy(model, testing_data_per_view[i], testing_labels_per_view[i])

        print '| Accuracy for view {}: {:.3f}'.format(i + 1, accuracy) + ', ' + param_msg

    print '|'

if __name__ == '__main__':

    data_directory = 'data/speech'

    # If provided a command-line argument, use that as the data location.
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]

    data_locations, file_idx_location = util.find_file_locations(data_directory)
    
    data_list = list()
    
    for i in range(len(data_locations)):
        data_list.append(sio.loadmat(data_locations[i]))
    
    file_idx = sio.loadmat(file_idx_location)

    # Print info
    print ('\n'
           'Data Directory:       {}\n'
           'Classification Model: {}\n'
           'Using Full Phones:    {}\n'
    ).format(data_directory, classification_model, use_full_phones)

    for fold_number in [1, 2, 3, 4, 5]:
        runSingleFold(data_list, file_idx, fold_number)


