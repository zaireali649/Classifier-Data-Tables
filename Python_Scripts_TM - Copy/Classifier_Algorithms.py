# -*- coding: utf-8 -*-
"""
6/6/2017

Thatyana


For each defintion:
    
printChart(f, clf): Prints the following to a CSV file
                    Table with Precision, Recall, f1, and Support
                    Avg/total
                    Confusion Matrix
                    Accuracy

 
"""

import sys

from numpy import genfromtxt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import LeaveOneLabelOut


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


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


def printChart(f, clf):
    
    ## When an array is too large, numpy prints the corners of the array and 
    ## prints ... for the center. For displaying purposes, these statements
    ## print the entire array and remove the periods 
    n.set_printoptions(threshold='nan')
    confMat = n.zeros((9,9), dtype=n.int)   
    allPredictions = []
    allClassifications = []


    ## Creates table of classification and confusion matrix
    for train, test in loso:
        clf.fit(features[train], classifications[train])
        predictions = clf.predict(features[test])
        confMat = confMat + confusion_matrix(classifications[test],predictions)
        allPredictions.append(predictions)
        allClassifications.append(classifications[test])

    ## Connects all predictions and classifications
    allPredictions = n.concatenate(allPredictions)
    allClassifications = n.concatenate(allClassifications)
    
    
    # Classification Report returns a string that contains the chart, but
    # each element is not in a separate cell. Therefore, classRep splits
    # the string into an array of each of the elements, with 2 spaces as
    # the delimiter. 
    classReport = classification_report(allClassifications,allPredictions)
    classRep = classReport.split("  ")
    
    # Modified the avg/total line so it lines up with the rest of the chart.
    classRep[3] = ",  "
    classRep[153] = ": "
        
    # Neatly displays the elements into individual cells
    for i in classRep:
        i = i + ','
        if len(i) == 1:
            continue
        else:
            f.write(i)
    
    f.write('\n')
        
    ## Writes a readable confusion matrix into the csv file 
    ## (writing it as-is to the file just displays illegal characters)
    ## Also displays Accuracy calculated from classifications and predictions
    confMat = n.array2string(confMat, separator=', ')
    f.write(confMat.replace('[', '').replace(']','')) #removes the brackets created when making a numpy array
    f.write("\n\nAccuracy: " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
    

# function uses logistic regression classifier 
def logRegression(C, penalty, tolerance):
    
     ## Creates CSV file with Classifier type, and parameter values (C, Penalty, and Tolerance in that order)
     filetype = 'LogisticRegressionClassifier_%s_%s_%s.csv' % (str(C), str(penalty), str(tolerance))
     f = open(filetype, "w")
     filename = 'Logistic Regression\n' + 'C = ' + str(C) + '\nPenalty = ' + penalty + '\nTolerance = ' + str(tolerance) + '\n\n'
     f.write(filename)    
    
     ## Initializes classifier type, since all 3 of 
     ## the parameters the function takes are strings
     cValue = int(C)
     tolValue = float(tolerance)
     
     ## Sets classifier
     clf = LogisticRegression(C=cValue,penalty=penalty, tol=tolValue)
      
     printChart(f, clf)     
            
     ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
     print("\nData is now in \"LogisticRegressionClassifier_%s_%s_%s.csv\"\n" % (str(C), str(penalty), str(tolerance)))       
                

# function uses support vector machine with a linear kernel
def svMachineLinear():
    
    ## Creates CSV file and make title of classifier used
    f = open("SVMClassifier_Linear.csv", "w")
    f.write("Support Vector Machine: Linear Kernel\n\n")
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='linear')
     
    printChart(f, clf)
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"SVMClassifier_Linear.csv\"\n")  
    
    
# function uses support vector machine with a gaussian/rbf kernel
def svMachineRBF():
  
    ## Creates CSV file and makes title of classifier used
    f = open("SVMClassifier_RBF.csv", "w")
    f.write("Support Vector Machine: Gaussian/RBF Kernel\n\n")
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='rbf')
        
    printChart(f, clf)        
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"SVMClassifier_RBF.csv\"\n")    
    
# function uses Decision Tree classifier
def decTree():
    
    ## Create CSV file and make fancy title
    f = open("DecisionTreeClassifier.csv", "w")
    f.write("Decision Tree\n\n")
    
    ## Classifier type for this is Decision Tree (note, it's considerably faster 
    ## than the other classifiers, in that the results are written sooner)
    clf = tree.DecisionTreeClassifier()
    
    printChart(f, clf)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"DecisionTreeClassifer.csv\"\n")     


# function uses AdaBoost classifier which takes in a classifier type
# and number of estimators as two strings. 
def adaBoost(baseEstimator, nEst):
   
    ## number of estimators comes into the function as a string
    ## so it must be converted to int before it can pass through 
    ## the classifier parameter
    nEstimators = int(nEst)    
    
    ## When a user enters --ada on the command line, these are 
    ## the options for the third argument this specific function
    ## can take. AdaBoost has a parameter called base_estimator
    ## which is the classifier the user chooses to use for the data
    if baseEstimator == 'dtc':
        baseEst = 'DecisionTreeClassifier'
        clf = AdaBoostClassifier(tree.DecisionTreeClassifier(), n_estimators=nEstimators, algorithm='SAMME')
        
    elif baseEstimator == 'rfc':
        baseEst = 'RandomForestClassifier'
        clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=nEstimators, algorithm='SAMME')
        
    elif baseEstimator == 'perc':
        baseEst = 'Perceptron'
        clf = AdaBoostClassifier(Perceptron(), n_estimators=nEstimators, algorithm='SAMME')
    
    ## After getting the necessary information, a unique file
    ## is opened, with AdaBoost followed by the type of classifier
    ## and number of estimators in the file name.       
    filetype = 'AdaBoostClassifier_%s_%s.csv' % (baseEst, nEst)
    f = open(filetype, "w")
    filename = ('AdaBoost with %s\nN Estimators = %s\n\n' % (baseEst, nEst))
    f.write(filename)
    printChart(f, clf)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it   
    filer = '\nData is now in AdaBoostClassifier_%s_%s.csv' % (baseEst, nEstimators)
    print(filer)    


def main():

    clfSelect = sys.argv[1]  
    
    ## Dictionary to hold classifier function
    classifierDict = {'1':logRegression, '2':svMachineLinear, '3':svMachineRBF, '4':decTree, '5':adaBoost}


    if clfSelect == '--logr':
        if len(sys.argv) == 2:
            classifierDict['1'](10, 'l1', .01)
        else:
            classifierDict['1'](sys.argv[2], sys.argv[3], sys.argv[4])
        
    elif clfSelect == '--svml':
        classifierDict['2']()    
            
    elif clfSelect == '--svmrbf':
        classifierDict['3']()    
                
    elif clfSelect == '--dectree':
        classifierDict['4']()
                    
    elif clfSelect == '--ada':
        if len(sys.argv) == 2:
            classifierDict['5']('DecisionTreeClassifier', 50)
        else:
            classifierDict['5'](sys.argv[2], sys.argv[3])
                        
    elif clfSelect == '--all': #this option will just run every classifier with default values
        classifierDict['1']()
        classifierDict['2']()
        classifierDict['3']()
        classifierDict['4']()
        classifierDict['5']()
                                                      
    else:
        exit(0)  #any key to exit    

if __name__ == '__main__':
   main()