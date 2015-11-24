'''
DataPreProcessor.py

Class for pre-processing data for GCCA

'''

import numpy as np
import scipy.io as sio

class DataPreProcessor:
    
    def __init__(self, data_locations, file_idx_location, blocks):
        '''
        Constructor for DataPreProcessor.

        Args:
            data_locations (list<str>): Directories for each view's data
            file_idx_location (str): Directory for file containing partitioning information
            blocks (int): Indices of file blocks to be used for training
        '''
        
        self.data_locations = data_locations
        self.file_idx_location = file_idx_location
        self.blocks = blocks
    
    def process(self):
        vocabulary_per_view = list()
        mfcc_per_view = list()
        labels_per_view = list()
        
        file_idx_contents = sio.loadmat(self.file_idx_location)
        file_blocks = file_idx_contents['files'][0]
        
        number_of_views = len(self.data_locations)
        
        for i in range(number_of_views):
            # Load .mat files for each view
            data_contents = sio.loadmat(self.data_locations[i])
            
            # Load relevant variables
            words = data_contents['Words'][0]
            valid_files = data_contents['Valid_Files'][0]
            frame_locs = data_contents['frame_locs'][0]
            
            mfcc_per_view.append(data_contents['MFCC'])
            labels_per_view.append(data_contents['Phones'][0])
            
            vocabulary = dict()
            
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
                    frame_loc = frame_locs[j] - 1 # convert to 0-based index
                    prev_frame_loc = 0
                    
                    if j > 0:
                        prev_frame_loc = frame_locs[j-1]
                        
                    # Start building a dictionary of words and store their positions within MFCC data
                    
                    uttered_words = words[prev_frame_loc:frame_loc]
                    number_of_words = len(uttered_words)
                    
                    if len(uttered_words) == 0:
                        continue
                    
                    current_word = uttered_words[0]
                    word_start_index = prev_frame_loc
                    word_end_index = prev_frame_loc
                    
                    for uttered_word in uttered_words:
                        if uttered_word[0] != current_word[0]:
                            vocabulary[current_word[0]] = (word_start_index, word_end_index - 1)
                            word_start_index = word_end_index
                            current_word = uttered_word
                        if word_end_index == number_of_words - 1:
                            vocabulary[current_word[0]] = (word_start_index, word_end_index)
                        word_end_index = word_end_index + 1
                    
            vocabulary_per_view.append(vocabulary)
            
        # Find common words across views             
        common_words = []
          
        reference_vocabulary = vocabulary_per_view[0]
        
        for word in reference_vocabulary.keys():
            is_common_word = True
            for i in range(1, len(vocabulary_per_view), 1):
                vocabulary = vocabulary_per_view[i]
                if word not in vocabulary:
                    is_common_word = False
                    break
            if is_common_word:
                common_words.append(word)
        
        # Extract corresponding MFCC data and perform dynamic time warping
        training_data_per_view = list()
        training_labels_per_view = list()
        
        for i in range(number_of_views): # Initialize with empty arrays
            training_data_per_view.append(np.ndarray(shape=(np.shape(mfcc_per_view[i])[0], 0), dtype=np.float))
            training_labels_per_view.append(np.array([]))
        
        for common_word in common_words:
            mfcc_list = list()
            label_list = list()
            frame_size_dict = dict()
            
            for i in range(number_of_views):
                vocabulary = vocabulary_per_view[i]
                word_loc = vocabulary[common_word]
                
                mfcc = mfcc_per_view[i]
                mfcc = mfcc[:,word_loc[0]:word_loc[1]]
                mfcc_list.append(mfcc)
                
                labels = labels_per_view[i]
                label_list.append(labels[word_loc[0]:word_loc[1]])
                
                frame_size_dict[i] = np.shape(mfcc)[1]
                
            sorted_indices = sorted(frame_size_dict, key=frame_size_dict.get, reverse=False)
            
            ref_mfcc_index = sorted_indices[int(len(sorted_indices) / 2)]
            
            for i in range(number_of_views):
                warped_data, warped_labels = self.warpTimeFrame(mfcc_list[i], mfcc_list[ref_mfcc_index], label_list[i])
                training_data_per_view[i] = np.hstack((training_data_per_view[i], warped_data))
                training_labels_per_view[i] = np.hstack((training_labels_per_view[i], warped_labels))
        
        for i in range(number_of_views):
            training_data_per_view[i] = self.center(training_data_per_view[i])
        
        return training_data_per_view, training_labels_per_view
    
    def warpTimeFrame(self, warp_matrix, ref_matrix, labels):
        '''
        Performs dynamic time warping
        
        References the following code:
            https://github.com/perivar/FindSimilar/blob/master/MatchBox/matlab/dtw.m

        '''
        
        warp_matrix_cols = np.shape(warp_matrix)[1]
        ref_matrix_cols = np.shape(ref_matrix)[1]
        
        if ref_matrix_cols == warp_matrix_cols:
            return warp_matrix, labels
        
        d = np.ndarray(shape=(warp_matrix_cols, ref_matrix_cols), dtype=float)
        
        for i in range(warp_matrix_cols):
            for j in range(ref_matrix_cols):
                feature_distances = warp_matrix[:, i] - ref_matrix[:, j]
                d[i][j] = np.sqrt(np.sum(np.power(feature_distances, 2)))
        
        g = np.ndarray(shape=(warp_matrix_cols + 1, ref_matrix_cols + 1), dtype=float)
        
        for i in range(warp_matrix_cols + 1):
            for j in range(ref_matrix_cols + 1):
                if i == 0 and j == 0:
                    g[i][j] = 2 * d[1][1]
                else:
                    g[i][j] = float("inf")
        
        for i in range(1, warp_matrix_cols + 1):
            for j in range(1, ref_matrix_cols + 1):
                i_l = i - 1
                j_l = j - 1
                
                min_distance = g[i][j-1] + d[i_l][j_l]
                
                distance = g[i-1][j-1] + 2 * d[i_l][j_l]
                
                if distance < min_distance:
                    min_distance = distance
                    
                distance = g[i-1][j] + d[i_l][j_l]
                
                if distance < min_distance:
                    min_distance = distance
                
                g[i][j] = min_distance
                
        warped_data = np.ndarray(shape=(np.shape(warp_matrix)[0], ref_matrix_cols), dtype=float)
        warped_labels = np.ndarray((ref_matrix_cols,), dtype=object)
        
        for j in range(1, ref_matrix_cols + 1):
            min_value = float("inf")
            min_index = 0
                
            for i in range(1, warp_matrix_cols + 1):
                if g[i][j] < min_value:
                    min_value = g[i][j]
                    min_index = i - 1
                
            warped_data[:,j-1] = warp_matrix[:,min_index]
            warped_labels[j-1] = labels[min_index]
            
        return warped_data, warped_labels

    def center(self, data):
        '''
        Centers data to zero-mean.
        Args:
            data (numpy.ndarray): Data to center.
        Returns:
            numpy.ndarray, the normalized data.
        '''

        out = np.array(data, copy=True, dtype='float')

        mean = np.mean(out, axis=1)
        for x in out.T:
            x -= mean
        return out
        
if __name__ == '__main__':
    # Test dynamic time warping
    
    A = np.array([[4, 1, 1, 1, 4]], np.float)
    B = np.array([[4, 1, 4]], np.float)
    
    processor = DataPreProcessor(None, None, None)
    
    warped_data = processor.warpTimeFrame(B, A)
    print warped_data
    
    warped_data = processor.warpTimeFrame(A, B)
    print warped_data
    
    A = np.array([[5, 3, 9, 7, 3]], np.float)
    B = np.array([[4, 7, 4]], np.float)
    
    warped_data = processor.warpTimeFrame(B, A)
    print warped_data
    
    warped_data = processor.warpTimeFrame(A, B)
    print warped_data
