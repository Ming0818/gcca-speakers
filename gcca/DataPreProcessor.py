'''
DataPreProcessor.py

Class for pre-processing data for GCCA

'''

import numpy as np
import scipy.io as sio

class DataPreProcessor:
    
    def __init__(self, data_locations, file_idx_locations, blocks):
        '''
        Constructor for DataPreProcessor.

        Args:
            data_locations (list<str>): Directories for each view's data
            file_idx_locations (list<str>): Directories for files containing partitioning information
            blocks (int): Indices of file blocks to be used for training
        '''
        
        self.data_locations = data_locations
        self.file_idx_locations = file_idx_locations
        self.blocks = blocks
    
    def process(self):
        vocabulary_per_view = dict()
        
        for i in range(len(self.data_locations)):
            # Load .mat files for each view
            data_contents = sio.loadmat(self.data_locations[i])
            file_idx_contents = sio.loadmat(self.file_idx_locations[i])
            
            # Load relevant variables
            words = data_contents['Words'][0]
            valid_files = data_contents['Valid_Files'][0]
            frame_locs = data_contents['frame_locs'][0]
            file_blocks = file_idx_contents['files'][0]
            
            vocabulary = list()
            
            # Create vocabulary for relevant blocks
            for j in range(len(valid_files)):
                valid_file = valid_files[j]
                
                # Check if this file falls under a valid block
                is_valid_file = False
                
                for block in self.blocks:
                    if valid_file in file_blocks[block-1]:
                        is_valid_file = True
                
                # If valid, then add words (corresponding to this file) to vocabulary 
                if is_valid_file:
                    frame_loc = frame_locs[j] - 1
                    prev_frame_loc = 0
                    
                    if j > 0:
                        prev_frame_loc = frame_locs[j-1]
                    
                    vocabulary.append(words[prev_frame_loc:frame_loc])
                    
            vocabulary_per_view[i] = vocabulary
            
            # Find common words across views               
                
                        
