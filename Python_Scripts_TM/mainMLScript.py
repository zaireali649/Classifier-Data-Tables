# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 18:36:50 2014

@author: Johnny
"""


from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import LeaveOneLabelOut


from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


import numpy as n
import numpy as np
import featureExtraction as featExtr

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def matrix(y, n):
    np.set_printoptions(threshold='nan')
    m = np.zeros((len(y), n), dtype=np.int)
    for x in range(len(y)):
        m[x][y[x]] = 1   
    return (m)    
    
def replace(y):
    w = set(y)
    w = list(w)
#    print w
    for x in range(len(set(y))):
        if (w[x] != x):
            for z in range(len(y)):
                if (y[z] == w[x]):
                    y[z] = x
    w = set(y)
    w = list(w)
#    print w
    return y
    

# File IO
path = 'rawData128SinglePoint.csv'
data = genfromtxt(path, delimiter=',', skip_header=1,usecols=range(0,384))
patientID = genfromtxt(path, delimiter=',', skip_header=1,usecols=[384])
classifications = genfromtxt(path, delimiter=',', skip_header=1,usecols=[385])



# Feature Extraction Script Call
features = featExtr.main(data,classifications)

# Subsetting
features = features[:,:]



# Cross- Validation Schemes
kf = KFold(len(classifications), 10)
loso = LeaveOneLabelOut(patientID)


# Classifier Selection
#clf = LDA()
#clf = LogisticRegression(C=10,penalty='l1', tol=.1) #*****
#clf = KNeighborsClassifier(n_neighbors=3)
#clf = svm.SVC(kernel="linear") #****
clf = svm.SVC(kernel="rbf") #****
#clf = tree.DecisionTreeClassifier()
#clf = GaussianNB()
#clf = RandomForestClassifier(max_depth=35, n_estimators=1000, max_features=15)
#clf = AdaBoostClassifier(n_estimators=1000) *****

name = 'SVM RBF'

# Needed for RBF SVM
features = preprocessing.scale(features)


np.set_printoptions(threshold='nan')
confMat = n.zeros((9,9))   
allPredictions = []
allClassifications = []



for train, test in loso:
    clf.fit(features[train], classifications[train])
    predictions = clf.predict(features[test])
    confMat = confMat + confusion_matrix(classifications[test],predictions)
    allPredictions.append(predictions)
    allClassifications.append(classifications[test])



allPredictions = n.concatenate(allPredictions)
allClassifications = n.concatenate(allClassifications)





#
#print(classification_report(allClassifications,allPredictions))
#print(confMat)
#print("Accuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))

allPredictions = replace(allPredictions)
allClassifications = replace(allClassifications)

score = matrix(allClassifications, len(set(allClassifications)))
test = matrix(allPredictions, len(set(allPredictions)))







def main():
    n_classes = len(set(classifications))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test.ravel(), score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
           
        
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
    mean_tpr /= n_classes
        
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
              
    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-Avg ROC (area = {0:0.6f})'
             ''.format(roc_auc["micro"]),
                      color='deeppink', linestyle=':', linewidth=4)
              
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-Avg ROC (area = {0:0.6f})'
             ''.format(roc_auc["macro"]),
                      color='navy', linestyle=':', linewidth=4)
              
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'gray', 'black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Class {0} ROC (area = {1:0.6f})'
                 ''.format(i, roc_auc[i]))
                  
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for %s' % (name))
    plt.legend(loc="lower right")
    plt.show()
        
#    plt.figure()
#    lw = 2
#    plt.plot(fpr[2], tpr[2], color='darkorange',
#             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
#    plt.legend(loc="lower right")
#    plt.show()   
        

if __name__ == '__main__':
   main()

