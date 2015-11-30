'''
test_baseline.py

Tests classification with SVM/kNN on raw data and with PCA/KPCA to serve as a
baseline for comparison with GCCA.

'''

from gcca import *
import util

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import os, sys, time


subsampling_factor = 1
vowel_labels = [0, 1, 3, 10, 17, 24, 33]
use_full_phones = True
use_cached = True

try_knn = True
try_svm = True


def format_time(t):
    return '{}m,{}s'.format(int(t)/60, int(t)%60)


def load_data(data_locations, file_idx_location, base_directory='data/cached/'):
    '''
    Loads data from each of the five blocks.

    Args:
        data_locations (list<str>): Locations to the data files.
        file_idx_location (str): Location of the fileidx.mat file.
    Returns:
        list<numpy.ndarray>, list of the data from different blocks.
        list<numpy.ndarray>, list of the labels from different blocks.
    '''

    if not base_directory.endswith('/'):
        base_directory = base_directory + '/'

    all_data = list()
    all_labels = list()

    try:
        if use_cached and os.path.exists(base_directory):

            for block in range(1, 5+1):

                data = np.load(base_directory+'data-{}.npy'.format(block))
                labels = np.load(base_directory+'labels-{}.npy'.format(block))

                all_data.append(data)
                all_labels.append(labels)

            return all_data, all_labels

    except Exception, error:
        print error
        print 'Unable to load cached files. Loading from .mat...'

    # Load from .mat files if not cache

    for block in range(1, 5+1):

        data_preprocessor = DataPreProcessor(data_locations, file_idx_location,
                [block], False)
        data, labels = data_preprocessor.process()

        for i in range(0, len(data)):
            data[i]   = np.array(data[i], dtype=np.float).T
            labels[i] = np.array(labels[i], dtype=np.int)

        if use_cached:
            if not os.path.exists('data/cached'):
                os.makedirs('data/cached')

            data_filename   = 'data-{}'.format(block)
            labels_filename = 'labels-{}'.format(block)

            np.save(base_directory+data_filename, data)
            np.save(base_directory+labels_filename, labels)

        all_data.append(data)
        all_labels.append(labels)

    return all_data, all_labels


def knn_accuracy(trn_data, trn_labels, tst_data, tst_labels, k_neighbors):

    knn = KNeighborsClassifier(k_neighbors)
    knn.fit(trn_data, trn_labels)
    results = knn.predict(tst_data)

    return np.sum(tst_labels == results)/float(tst_labels.size)


def tune_knn(trn_data, trn_labels, tun_data, tun_labels):

    best_k = 0
    best_acc = 0

    # Params empirically determined...
    for k in [25]:

        acc = knn_accuracy(trn_data, trn_labels, tun_data, tun_labels, k)

        if acc > best_acc:
            best_acc = acc
            best_k = k

    return best_k, best_acc


def svm_accuracy(trn_data, trn_labels, tst_data, tst_labels, gamma, error):

    svm = SVC(
        decision_function_shape='ovo',
        kernel='rbf',
        C=error,
        gamma=gamma,
    )
    svm.fit(trn_data, trn_labels)
    results = svm.predict(tst_data)

    return np.sum(tst_labels == results)/float(tst_labels.size)


def tune_svm(trn_data, trn_labels, tun_data, tun_labels):

    best_gamma = 0
    best_error = 0
    best_acc = 0

    # Params epirically determined...
    for gamma in range(10, 11):
        gamma *= 1e-8
        for error in [2.0]:

            acc = svm_accuracy(trn_data, trn_labels, tun_data, tun_labels, gamma,
                    error)

            if acc > best_acc:
                best_acc = acc
                best_gamma = gamma
                best_error = error

    return best_gamma, best_error, best_acc


def evaluate(all_data, all_labels, fold_number, view_number):
    '''
    Runs baseline classification for a single fold.

    Args:
        all_data (list<numpy.ndarray>): List of all data from each block.
        all_labels (list<numpy.ndarray>): List of all labels from each block.
        fold_number (int): The fold to test on.
    '''

    print '---- Fold #{}, View #{} ----'.format(fold_number, view_number)

    def report_time(t):
        print '\tTook {}\n'.format(format_time(time.time()-t))

    # Find the appropriate blocks for this fold and stack data

    ( trn_blocks,
      tun_blocks,
      tst_blocks ) = util.configure_blocks(fold_number)

    def data_for_blocks(blocks):

        data = list()
        labels = list()

        for block in blocks:
            data.append(all_data[block-1][view_number][0::subsampling_factor,:])
            labels.append(all_labels[block-1][view_number][0::subsampling_factor])

        return np.vstack(data), np.hstack(labels)

    trn_data, trn_labels = data_for_blocks(trn_blocks)
    tun_data, tun_labels = data_for_blocks(tun_blocks)
    tst_data, tst_labels = data_for_blocks(tst_blocks)

    '''
    # Make data zero-mean
    mean = np.mean(trn_data, axis=0)
    trn_data = trn_data-mean
    tun_data = tun_data-mean
    tst_data = tst_data-mean
    '''

    print (
        'Num. samples in training: {}\n'
        'Num. samples in tuning:   {}\n'
        'Num. samples in testing:  {}\n'
    ).format(trn_data.shape[0], tun_data.shape[0], tst_data.shape[0])



    def run(trn, tun, tst):

        if try_knn:
            print 'kNN:'

            runtime = time.time()

            k_neighbors, tun_acc = tune_knn(
                    trn, trn_labels,
                    tun, tun_labels
            )
            tst_acc = knn_accuracy(
                    trn, trn_labels,
                    tst, tst_labels,
                    k_neighbors
            )
            print (
                '\tk-Neighbors:      {}\n'
                '\tTuning Accuracy:  {:.2f}%\n'
                '\tTesting Accuracy: {:.2f}%\n'
            ).format(k_neighbors, 100*tun_acc, 100*tst_acc)

            report_time(runtime)

        if try_svm:
            print 'SVM:'

            runtime = time.time()

            gamma, error, tun_acc = tune_svm(
                    trn_data, trn_labels,
                    tun_data, tun_labels
            )
            tst_acc = svm_accuracy(
                    trn_data, trn_labels,
                    tst_data, tst_labels,
                    gamma, error
            )
            print (
                '\tGamma:            {}\n'
                '\tPenalty:          {}\n'
                '\tTuning Accuracy:  {:.2f}%\n'
                '\tTesting Accuracy: {:.2f}%\n'
            ).format(gamma, error, 100*tun_acc, 100*tst_acc)

            report_time(runtime)

    print 'Raw data...\n'

    run(trn_data, tun_data, tst_data)

    print 'PCA...\n'

    pca = PCA()
    pca.fit(trn_data)

    trn_data_pca = pca.transform(trn_data)
    tun_data_pca = pca.transform(tun_data)
    tst_data_pca = pca.transform(tst_data)

    run(trn_data_pca, tun_data_pca, tst_data_pca)


if __name__ == '__main__':

    data_directory = 'data/speech/'

    # If provided a command-line argument, use that as the data location
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]

    data_locations, file_idx_location = util.find_file_locations(data_directory)

    # Print info
    print (
        '\n'
        'Data Directory:     {}\n'
        'Subsampling Factor: {}\n'
    ).format(data_directory, subsampling_factor)

    all_data, all_labels = load_data(data_locations, file_idx_location)

    for fold in range(0+1, 5+1):
        for view in range(0, 4):
            evaluate(all_data, all_labels, fold, view)


