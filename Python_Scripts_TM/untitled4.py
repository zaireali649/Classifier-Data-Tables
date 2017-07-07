# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:10:37 2017

@author: morales
"""

import numpy as np
from sklearn.cross_validation import KFold
from numpy import genfromtxt
import featureExtraction as featExtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


path = 'User0SinglePointData.csv'
data = genfromtxt(path, delimiter=',', skip_header=1,usecols=range(0,384))
patientID = genfromtxt(path, delimiter=',', skip_header=1,usecols=[384])
classifications = genfromtxt(path, delimiter=',', skip_header=1,usecols=[385])

# Feature Extraction Script Call
features = featExtr.main(data,classifications)

# Subsetting
features = features[:,:]

# Split into train/test

# Cross- Validation Schemes
kf = KFold(len(classifications), 10, shuffle=True)
clf = LogisticRegression(C=10,penalty='l1', tol=.1) 
allPredictions = []
allClassifications = []

for train, test in kf:        
    clf.fit(features[train], classifications[train])
    predictions = clf.predict(features[test])
#    confMat = confMat + confusion_matrix(classifications[test],predictions)
    allPredictions.append(predictions)
    allClassifications.append(classifications[test])
        
    
    
print(set(classifications))    
allPredictions = np.concatenate(allPredictions)
allClassifications = np.concatenate(allClassifications)


print(classification_report(allClassifications,allPredictions))