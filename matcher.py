from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

def knn(X, y):
    knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan') 
    gen_scores = []
    imp_scores = []
    
    for i in range(0, len(y)):
        query = X[i, :]
        query_label = y[i]
        
        templates = np.delete(X, i, 0)
        template_labels = np.delete(y, i)
            
        # Set the appropriate labels
        # 1 is genuine, 0 is impostor
        y_hat = np.zeros(len(template_labels))
        y_hat[template_labels == query_label] = 1 
        y_hat[template_labels != query_label] = 0
        
        knn.fit(templates, y_hat) # Train the classifier
        scores = knn.predict_proba(query.reshape(1,-1)).reshape(1,2) # Predict the label of the query
        classes = knn.classes_.reshape(1,2)
        
        gen_scores.extend(scores[classes==1])
        imp_scores.extend(scores[classes==0])
        
    return gen_scores, imp_scores

def nb(X, y):
        gnb = GaussianNB() 

        labels_correct = []
        labels_incorrect = []

        for i in range(0, len(y)):
            query = X[i, :]
            query_label = y[i]
        
            template_imgs = np.delete(X, i, 0)
            template_labels = np.delete(y, i)
            
            # Set the appropriate labels
            # 1 is genuine, 0 is impostor
            y_hat = np.zeros(len(template_labels))
            y_hat[template_labels == query_label] = 1 
            y_hat[template_labels != query_label] = 0
        
            gnb.fit(template_imgs, y_hat) # Train the classifier
            scores = gnb.predict_proba(query.reshape(1,-1)).reshape(1,2) # Predict the label of the query
            classes = gnb.classes_.reshape(1,2)
        
            labels_correct.extend(scores[classes==1])
            labels_incorrect.extend(scores[classes==0])
                
        return labels_correct, labels_incorrect