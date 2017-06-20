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
import featureExtraction as featExtr



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
clf = LogisticRegression(C=10,penalty='l1', tol=.01) #*****
#clf = KNeighborsClassifier(n_neighbors=3)
#clf = svm.SVC(kernel="linear") ****
#clf = tree.DecisionTreeClassifier() *****
#clf = GaussianNB()
#clf = RandomForestClassifier(max_depth=35, n_estimators=1000, max_features=15)
#clf = AdaBoostClassifier(n_estimators=1000) *****



# Needed for RBF SVM
#features = preprocessing.scale(features)



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
print(classification_report(allClassifications,allPredictions))
print(confMat)
print("Accuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))