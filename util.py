'''
util.py

Common utility module used for driver scripts.

'''

import matplotlib.cm as cmx
import matplotlib.colors as colors

import os


# Enum class to specify possible classifiers to use
class ClassificationModel:
    K_Neighbors = 'K Nearest Neighbors'
    Kernel_SVM_RBF = 'Kernel SVM - RBF'


def configure_blocks(fold_number, number_of_folds=5):
    '''
    Configure blocks depending on fold.

    Args:
        fold_number (int): The number of the fold to configure it for.
    Args (optional):
        number_of_folds (int): The total number of potential folds.
    Returns:
        list<int>, the blocks to be used for training.
        list<int>, the blocks to be used for tuning.
        list<int>, the blocks to be used for testing.
    '''

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

    return training_blocks, tuning_blocks, testing_blocks


def find_file_locations(data_directory):
    '''
    Finds data locations for the .mat files in a given directory.

    Args:
        data_directory (str): The directory where the data lives.
    Returns:
        list<str>, the locations of the .mat files for each speaker.
        str, the location of the fileidx.mat file.
    '''

    data_locations = list()
    file_idx_location = None

    try:
        for file in os.listdir(data_directory):
            if file.startswith('JW') and file.endswith('.mat'):
                data_locations.append(os.path.join(data_directory, file))
            if file.endswith('fileidx.mat'):
                file_idx_location = os.path.join(data_directory, file)
    except:
        raise RuntimeError('Unable to find data files in {}'.format(
            data_directory))

    data_locations.sort()

    return data_locations, file_idx_location


def getColorMap(N):
    '''
    Returns a function that maps each index in [0,1,...,N-1] to a distinct
    RGB color.
    '''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


