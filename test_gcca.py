'''
Driver for performing generalized CCA on speech data

'''

from gcca.GeneralizedCCA import GeneralizedCCA

import os

if __name__ == '__main__':
    data_directory = 'data/speech/'
    
    data_locations = list()
    
    for file in os.listdir(data_directory):
        if file.endswith('.mat'):
            data_locations.append(os.path.join(data_directory, file))
    
    model = GeneralizedCCA(data_locations, 10)
    
    model.solve()
    