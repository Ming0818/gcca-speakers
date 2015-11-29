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


