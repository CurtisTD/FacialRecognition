import matplotlib.pyplot as plt
import numpy as np
#from sklearn.datasets import fetch_olivetti_faces
import get_data

def get_lbp(image, width):
    image = image.reshape((width, width))
    lbp_image = np.zeros(shape=(width, width))
    num_neighbors = 1
    
    for i in range(num_neighbors, image.shape[0] - num_neighbors):
        for j in range(num_neighbors, image.shape[1] - num_neighbors):
            center_pixel = image[i,j]
            binary_string = ""
            
            for m in range(i-num_neighbors, i+num_neighbors+1):
                for n in range(j-num_neighbors, j+num_neighbors+1):
                    
                    if [i,j] == [m,n]: # same pixel
                        pass
                    else:
                        neighbor_pixel = image[m, n]                                           
                        if center_pixel >= neighbor_pixel:
                            binary_string += '1'
                        else:
                            binary_string += '0'
            
            lbp_image[i,j] = int(binary_string, 2)            
    return lbp_image

def get_features(image, size):
    # compute and concatenate histograms
    histograms = []
    for i in range(0,image.shape[0],size):
        for j in range(0,image.shape[1],size):
            block = image[i:i+size, j:j+size]
            histograms.extend( np.histogram(block, bins=256)[0] )    
    return histograms

#########################################################################

