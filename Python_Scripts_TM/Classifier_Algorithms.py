# -*- coding: utf-8 -*-
"""
6/6/2017

Thatyana

Length of time to complete each (from shortest to longest)
1. Decision Tree (3-4 seconds)
2. SVM Linear (takes about 3-5 seconds)
3. SVM RBF (a little over 5 seconds)
4. Logistic Regression (takes about 20 seconds)
5. AdaBoost (takes like 8 minutes with n_estimators=1000)
 
"""


from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
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

#Needed for SVM RBF
features = preprocessing.scale(features)



# Cross- Validation Schemes
kf = KFold(len(classifications), 10)
loso = LeaveOneLabelOut(patientID)


# function uses logistic regression classifier 
# Using Jack's data, Accuracy: .923586
def logRegression():
    
    ## Create CSV file and make fancy title
    f = open("LogisticRegressionClassifier.csv", "w")
    f.write("\n###############################################################\n")
    f.write("Logistic Regression\n\n")
    
    ## Initializes confusion matrix and classifier type
    clf = LogisticRegression(C=10,penalty='l1', tol=.01)
    confMat = n.zeros((9,9))   
    allPredictions = []
    allClassifications = []

    ## Creates table of classification and confusion matrix
    for train, test in loso:
        clf.fit(features[train], classifications[train])
        predictions = clf.predict(features[test])
        confMat = confMat + confusion_matrix(classifications[test],predictions)
        allPredictions.append(predictions)
        allClassifications.append(classifications[test])

    ## Connects all predictions and classifications and writes the table to the csv file
    allPredictions = n.concatenate(allPredictions)
    allClassifications = n.concatenate(allClassifications)
    f.write(classification_report(allClassifications,allPredictions))
    
    ## Writes a readable confusion matrix into the csv file (writing it as-is to the file just displays illegal characters)
    ## Also displays Accuracy calculated from classifications and predictions
    f.close    
    with open("LogisticRegressionClassifier.csv", 'a') as f:
            f.write(n.array2string(confMat, separator=', '))
            f.write("\n\nAccuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
            f.write("\n###############################################################\n")
            
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("Data is now in \"LogisticRegressionClassifier.csv\"\n")        
                

# function uses support vector machine with a linear kernel
# As of now, Accuracy: .914769
def svMachineLinear():
    
    ## Create CSV file and make title of classifier used
    f = open("SVMClassifier_Linear.csv", "w")
    f.write("\n###############################################################\n")
    f.write("Support Vector Machine: Linear Kernel\n\n")
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='linear')
    confMat = n.zeros((9,9))   
    allPredictions = []
    allClassifications = []

    ## Creates table of classification and confusion matrix
    for train, test in loso:
        clf.fit(features[train], classifications[train])
        predictions = clf.predict(features[test])
        confMat = confMat + confusion_matrix(classifications[test],predictions)
        allPredictions.append(predictions)
        allClassifications.append(classifications[test])
        
    allPredictions = n.concatenate(allPredictions)
    allClassifications = n.concatenate(allClassifications)
    f.write(classification_report(allClassifications,allPredictions))
    
    
    ## Writes a readable confusion matrix into the csv file (writing it as-is to the file just displays illegal characters)
    ## Also displays Accuracy calculated from classifications and predictions
    f.close    
    with open("SVMClassifier_Linear.csv", 'a') as f:
            f.write(n.array2string(confMat, separator=', '))
            f.write("\n\nAccuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
            f.write("\n###############################################################\n") 
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("Data is now in \"SVMClassifier_Linear.csv\"\n")  
    
    
# function uses support vector machine with a gaussian/rbf kernel
# As of now, Accuracy: 0.927259
def svMachineRBF():

    
    ## Create CSV file and make title of classifier used
    f = open("SVMClassifier_RBF.csv", "w")
    f.write("\n###############################################################\n")
    f.write("Support Vector Machine: Gaussian/RBF Kernel\n\n")
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='rbf')
    confMat = n.zeros((9,9))   
    allPredictions = []
    allClassifications = []
    
    #This is CRUCIAL to SVM RBF. Without this, it throws a Metrics Error
    features1 = preprocessing.scale(features)
    
    ## Creates table of classification and confusion matrix
    for train, test in loso:
        clf.fit(features1[train], classifications[train])
        predictions = clf.predict(features1[test])
        confMat = confMat + confusion_matrix(classifications[test],predictions)
        allPredictions.append(predictions)
        allClassifications.append(classifications[test])
        
    allPredictions = n.concatenate(allPredictions)
    allClassifications = n.concatenate(allClassifications)
    f.write(classification_report(allClassifications,allPredictions))
    
    
    ## Writes a readable confusion matrix into the csv file (writing it as-is to the file just displays illegal characters)
    ## Also displays Accuracy calculated from classifications and predictions
    f.close    
    with open("SVMClassifier_RBF.csv", 'a') as f:
            f.write(n.array2string(confMat, separator=', '))
            f.write("\n\nAccuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
            f.write("\n###############################################################\n") 
            
            
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("Data is now in \"SVMClassifier_RBF.csv\"\n")    
    
# function uses Decision Tree classifier
# As of now, Accuracy: 0.756062
def decTree():
    
    ## Create CSV file and make fancy title
    f = open("DecisionTreeClassifier.csv", "w")
    f.write("\n###############################################################\n")
    f.write("Decision Tree\n\n")
    
    ## Classifier type for this is Decision Tree (note, it's considerably faster 
    ## than the other classifiers, in that the results are written sooner)
    clf = tree.DecisionTreeClassifier()
    confMat = n.zeros((9,9))   
    allPredictions = []
    allClassifications = []

    ## Creates table of classification and confusion matrix
    for train, test in loso:
        clf.fit(features[train], classifications[train])
        predictions = clf.predict(features[test])
        confMat = confMat + confusion_matrix(classifications[test],predictions)
        allPredictions.append(predictions)
        allClassifications.append(classifications[test])
        
    allPredictions = n.concatenate(allPredictions)
    allClassifications = n.concatenate(allClassifications)
    f.write(classification_report(allClassifications,allPredictions))    
    
    ## Writes a readable confusion matrix into the csv file (writing it as-is to the file just displays illegal characters)
    ## Also displays Accuracy calculated from classifications and predictions
    f.close    
    with open("DecisionTreeClassifier.csv", 'a') as f:
            f.write(n.array2string(confMat, separator=', '))
            f.write("\n\nAccuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
            f.write("\n###############################################################\n")
        
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("Data is now in \"DecisionTreeClassifer.csv\"\n")     


# function uses AdaBoost classifier
# Jack's data: Accuracy: 
def adaBoost():
   
    ## Create CSV file and make fancy title
    f = open("AdaBoostClassifier.csv", "w")
    f.write("\n###############################################################\n")
    f.write("AdaBoost (With DecisionTreeClassifier\n\n")
    
    ## Classifier type for this is AdaBoost
    clf = AdaBoostClassifier(n_estimators=50)
    confMat = n.zeros((9,9))   
    allPredictions = []
    allClassifications = []

    ## Creates table of classification and confusion matrix
    for train, test in loso:
        clf.fit(features[train], classifications[train])
        predictions = clf.predict(features[test])
        confMat = confMat + confusion_matrix(classifications[test],predictions)
        allPredictions.append(predictions)
        allClassifications.append(classifications[test])
        
    allPredictions = n.concatenate(allPredictions)
    allClassifications = n.concatenate(allClassifications)
    f.write(classification_report(allClassifications,allPredictions))
    
    
    ## Writes a readable confusion matrix into the csv file (writing it as-is to the file just displays illegal characters)
    ## Also displays Accuracy calculated from classifications and predictions
    f.close    
    with open("AdaBoostClassifier.csv", 'a') as f:
            f.write(n.array2string(confMat, separator=', '))
            f.write("\n\nAccuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
            f.write("\n###############################################################\n")
       
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("Data is now in \"AdaBoostClassifier.csv\"\n")     


# Beginning of program; lists and asks user for classifier type
print("1: Logistic Regression")
print("2: Support Vector Machine: Linear")
print("3. Support Vector Machine: Gaussian/RBF")
print("4: Decision Tree")
print("5: AdaBoost (With DecisionTreeClassifier)")
print("6: All\n")

clfSelect = input("Enter the desired classifier number, or any other key to exit.\n")
iterations = input("How many iterations?\n")

# Dictionary to hold classifier function
classifierDict = {'1':logRegression, '2':svMachineLinear, '3':svMachineRBF, '4':decTree, '5':adaBoost}

if clfSelect == 1:
    classifierDict['1']()

elif clfSelect == 2:
    classifierDict['2']()    

elif clfSelect == 3:
    classifierDict['3']()    
    
elif clfSelect == 4:
    classifierDict['4']()
    
elif clfSelect == 5:
    classifierDict['5']()

elif clfSelect == 6:
    classifierDict['1']()
    classifierDict['2']()
    classifierDict['3']()
    classifierDict['4']()
    classifierDict['5']()
    

else:
    exit(0)  #any key to exit    


