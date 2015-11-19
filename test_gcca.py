'''
Driver for performing generalized CCA on speech data

'''

from gcca.DataPreProcessor import DataPreProcessor
from gcca.GeneralizedCCA import GeneralizedCCA

import os

if __name__ == '__main__':
    data_directory = 'data/speech/'
    
    data_locations = list()
    file_idx_locations = list()
    
    for file in os.listdir(data_directory):
        if file.startswith('JW') and file.endswith('.mat'):
            data_locations.append(os.path.join(data_directory, file))
        if file.startswith('fileidxJW') and file.endswith('.mat'):
            file_idx_locations.append(os.path.join(data_directory, file))
    
    data_locations.sort()
    file_idx_locations.sort()
    
    file_blocks = [1, 2, 3] # i.e. Use these blocks for training
    
    data_pre_processor = DataPreProcessor(data_locations, file_idx_locations, file_blocks)
    data_pre_processor.process()
    