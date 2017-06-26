# -*- coding: utf-8 -*-
"""
6/6/2017

Thatyana

##############################################################################
#                                    Overview                                #
##############################################################################
For this script, the user goes into the command line (makes sure they're 
in the folder with the script) and types this:
    
python Classifier_Algorithms.py --<classifier_type> <any additional parameters> <plot>

This script supports running the following classifiers:
    Logistic Regression (Parameters: C, penalty, tolerance)
    SVM Linear
    SVM RBF
    Decision Tree
    AdaBoost (Parameters: base_estimator, n_estimators)
    
As for the command line, a user must enter the script name followed by a 
classifier of their choice:
    Logistic Regression: --logr
    SVM Linear: --svml
    SVM RBF: --svmrbf
    Decision Tree: --dectree
    AdaBoost: --ada 

##############################################################################
#                   Additional Parameters and Default Values                 #
##############################################################################
Currently, each classifier can have additional optional parameters. Solely 
typing the classifier name will produce results given by default values

    Logistic Regression has 3 additional parameters:
        C has a default value of 10
        Penalty has a default value of 'l1'
        Tolerance has a default value of 0.01
        
    AdaBoost has 2 additional parameters: **
        base_estimator has a default value of 'dtc' which is Decision Tree
        n_estimators has a default value of 50
        
** for AdaBoost, currently the available base_estimators are as follows:
    dtc: DecisionTreeClassifier
    rfc: RandomForestClassifier
    perc: Perceptron

##############################################################################
#                                    Note                                    #
##############################################################################
When giving commands on the command line, the user can either enter just a 
classifier, or the classifier and ALL of its parameters. For example, if a user
wished to use Logistic Regression and input custom values, they must enter values
for C, Penalty, and Tolerance. Same goes for AdaBoost (both the base estimators
and n estimators must be entered)

##############################################################################
#                                    Plot                                    #
##############################################################################
As of now, the user also has the option to plot ROC curves for the gestures for
each classifier. Whether or not the user decides to input additional parameters, 
he/she may also enter 'plot' as a parameter. That gives the user 4 options:
      Default Values/No Additional Parameters without ROC Curve plot
      Default Values/No Additional Parameters with ROC Curve plot
      Custom Values/Additional Parameters without ROC Curve plot
      Custom Values/Additional Parameters with ROC Curve plot
      
##############################################################################
#                                  Examples                                  #
##############################################################################    
 
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


import numpy as np
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



# This function is used to convert the allClassifications and allPredictions
# array to a y x n array with the gestures binarized
def matrix(y, n):
    np.set_printoptions(threshold='nan')
    m = np.zeros((len(y), n), dtype=np.int)
    for x in range(len(y)):
        m[x][int(y[x])] = 1   
    return (m)    
 
# Since the LRUD Points were converted to one Point action,
# the set went from 0 1 2 3 4 5 6 7 8 9 10 11 
# to 0 1 2 3 4 5 9 10 11. To make formatting neater, 
# this function converts the 2nd set to 
# 0 1 2 3 4 5 6 7 8      
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

def plotRoc(tpr, fpr, roc_auc, n_classes):
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
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def printRocCurve(f, score, test, isPlot):
    
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
    
    f.write('\nROC AUC')
    for i in range(len(roc_auc)-1):
        if i in roc_auc:
            f.write('\nClass: {0}, {1:0.6f}'
            ''.format(i, roc_auc[i]))
    f.close()        
        
    # if the user selected to plot the ROC curve on matlib, this is where
    # the plot function is called
    if isPlot == 'true': 
        plotRoc(tpr, fpr, roc_auc, n_classes)

def printChart(f, clf, isPlot):
    
    ## When an array is too large, numpy prints the corners of the array and 
    ## prints ... for the center. For displaying purposes, these statements
    ## print the entire array and remove the periods 
    np.set_printoptions(threshold='nan')
    confMat = np.zeros((9,9), dtype=np.int)   
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
    allPredictions = np.concatenate(allPredictions)
    allClassifications = np.concatenate(allClassifications)
    

    
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
    confMat = np.array2string(confMat, separator=', ')
    f.write(confMat.replace('[', '').replace(']','')) #removes the brackets created when making a numpy array
    f.write("\n\nAccuracy:, " + ("%.6f"%accuracy_score(allClassifications,allPredictions)))
    f.write('\n')
        
    allPredictions2 = replace(allPredictions)
    allClassifications2 = replace(allClassifications)

    score = matrix(allClassifications2, len(set(allClassifications2)))
    test = matrix(allPredictions2, len(set(allPredictions2)))
    
    printRocCurve(f, score, test, isPlot)
    

# function uses logistic regression classifier 
def logRegression(C, penalty, tolerance, isPlot):
    
     ## Creates CSV file with Classifier type, and parameter values (C, Penalty, and Tolerance in that order)
     filetype = 'LogisticRegressionClassifier_%s_%s_%s.csv' % (str(C), str(penalty), str(tolerance))
     f = open(filetype, "w")
     filename = 'Logistic Regression\n' + 'C, ' + str(C) + '\nPenalty, ' + penalty + '\nTolerance, ' + str(tolerance) + '\n\n'
     f.write(filename)    
    
     ## Initializes classifier type, since all 3 of 
     ## the parameters the function takes are strings
     cValue = int(C)
     tolValue = float(tolerance)
     
     ## Sets classifier
     clf = LogisticRegression(C=cValue,penalty=penalty, tol=tolValue)
      
     printChart(f, clf, isPlot)     
            
     ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
     print("\nData is now in \"LogisticRegressionClassifier_%s_%s_%s.csv\"\n" % (str(C), str(penalty), str(tolerance)))       
                

# function uses support vector machine with a linear kernel
def svMachineLinear(isPlot):
    
    ## Creates CSV file and make title of classifier used
    f = open("SVMClassifier_Linear.csv", "w")
    f.write("Support Vector Machine: Linear Kernel\n\n")
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='linear')
     
    printChart(f, clf, isPlot)
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"SVMClassifier_Linear.csv\"\n")  
    
    
# function uses support vector machine with a gaussian/rbf kernel
def svMachineRBF(isPlot):
  
    ## Creates CSV file and makes title of classifier used
    f = open("SVMClassifier_RBF.csv", "w")
    f.write("Support Vector Machine: Gaussian/RBF Kernel\n\n")
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='rbf')
        
    printChart(f, clf, isPlot)        
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"SVMClassifier_RBF.csv\"\n")    
    
# function uses Decision Tree classifier
def decTree(isPlot):
    
    ## Create CSV file and make fancy title
    f = open("DecisionTreeClassifier.csv", "w")
    f.write("Decision Tree\n\n")
    
    ## Classifier type for this is Decision Tree (note, it's considerably faster 
    ## than the other classifiers, in that the results are written sooner)
    clf = tree.DecisionTreeClassifier()
    
    printChart(f, clf, isPlot)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"DecisionTreeClassifer.csv\"\n")     


# function uses AdaBoost classifier which takes in a classifier type
# and number of estimators as two strings. 
def adaBoost(baseEstimator, nEst, isPlot):
   
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
    filename = ('AdaBoost with %s\nEstimators:, %s\n\n' % (baseEst, nEst))
    f.write(filename)
    
    printChart(f, clf, isPlot)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it   
    filer = '\nData is now in AdaBoostClassifier_%s_%s.csv' % (baseEst, nEstimators)
    print(filer)    


def main():

    clfSelect = sys.argv[1]  
    
    ## Dictionary to hold classifier function
    classifierDict = {'1':logRegression, '2':svMachineLinear, '3':svMachineRBF, '4':decTree, '5':adaBoost}

    # logistic regression
    if clfSelect == '--logr':
        if len(sys.argv) == 2: #default value of not inserting any other parameters and no plotting
            classifierDict['1'](10, 'l1', .01, 'false')
        elif len(sys.argv) == 3:
            if sys.argv[2] == 'plot': #default values with plot
                classifierDict['1'](10, 'l1', .01, 'true')
        else:
            if len(sys.argv) == 5:
                classifierDict['1'](sys.argv[2], sys.argv[3], sys.argv[4], 'false') #custom values no plot
            else:
                classifierDict['1'](sys.argv[2], sys.argv[3], sys.argv[4], 'true') #custom values with plot
    
    # SVM linear              
    elif clfSelect == '--svml':
        if len(sys.argv) == 2:
            classifierDict['2']('false') #svm linear no plot
        else:
            if sys.argv[2] == 'plot': #svm linear with plot
                classifierDict['2']('true')
       
    # SVM RBF    
    elif clfSelect == '--svmrbf':
        if len(sys.argv) == 2:
            classifierDict['3']('false') #svm gaussian no plot
        else:
            if sys.argv[2] == 'plot': #svm gaussian with plot
                classifierDict['3']('true')    
    
    # Decision Tree        
    elif clfSelect == '--dectree':
        if len(sys.argv) == 2:
            classifierDict['4']('false') #Decision tree no plot
        else:
            if sys.argv[2] == 'plot': #Decision tree with plot
                classifierDict['4']('true') 
      
    # AdaBoost              
    elif clfSelect == '--ada':
        if len(sys.argv) == 2:
            classifierDict['5']('dtc', 50, 'false') #default values no plot
        elif len(sys.argv) == 3:
            if sys.argv[2] == 'plot':
                classifierDict['5']('dtc', 50, 'true') #default values with plot
        elif len(sys.argv) == 4:
            classifierDict['5'](sys.argv[2], sys.argv[3], 'false') #custom values no plot
        else:
            if sys.argv[4] == 'plot':
                classifierDict['5'](sys.argv[2], sys.argv[3], 'true') #custom values with plot
                        
    elif clfSelect == '--all': #this option will just run every classifier with default values and no plot
        classifierDict['1'](10, 'l1', .01, 'false')
        classifierDict['2']('false')
        classifierDict['3']('false')
        classifierDict['4']('false')
        classifierDict['5']('dtc', 50, 'false')
                                                      
    else:
        exit(0)  #any key to exit    

if __name__ == '__main__':
   main()