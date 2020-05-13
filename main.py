import get_data
import matcher
import performance 
import features_lbp
import numpy as np
import matplotlib.pyplot as plt

for blockSize in range(4, 21, 4):
    
    print("Blocksize = %d" % blockSize)

    width = 50
    
    ''' Load the data and their labels '''
    image_directory = 'Project Data/Original Images'
    X, y = get_data.get_images(image_directory)
    
    #landmark_directory = 'data/landmarks'
    #X, y = get_data.get_landmarks('Project Data/Landmarks')
    
    
    ''' Get LBP components '''
    lbp_features = []
    for i in range(len(X)):
            lbp_face = features_lbp.get_lbp(X[i], width)
            
            if i == 0:
                plt.figure("LBP Block Size = %d" % blockSize)
                plt.imshow(lbp_face, cmap="gray")
            
            features = features_lbp.get_features(lbp_face, blockSize)
            lbp_features.append(features)        
    lbp_features = np.array(lbp_features)
    
    
    ''' Get PCA components '''
    # X = get_features.pca(X)
    
    ''' Matching with KNN'''
    # gen_scores, imp_scores = matcher.knn(X, y)
    genScoresKNN, impScoresKNN = matcher.knn(lbp_features, y)
    
    ''' Matching with NB '''
    # labels_correct, labels_incorrect = matcher.nb(X, y)
    genScoresNB, impScoresNB = matcher.nb(lbp_features, y)
    
    ''' Performance assessment KNN'''
    #performance.perf(gen_scores, imp_scores)
    #performance.perf(gen_scores1, imp_scores1)
    
    ''' Performance assessment NB '''
    performance.perf(genScoresKNN, impScoresKNN)
    performance.perf(genScoresNB, impScoresNB)
    
