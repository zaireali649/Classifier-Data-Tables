# -*- coding: utf-8 -*-
"""
Created Jun 26 2017

@author: Thatyana Morales

Dependant on Classifier_Algorithms, this script handles everything involving
printing the data to the CSV files as well as ROC calculations and ROC curve
plots. 

"""

import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from Classifier_Algorithms_AllUsers_Command import classifications, loso, features
import relabelGestureSet as relabel


# This function is used to convert the allClassifications and allPredictions
# array to a y x n array with the gestures binarized
def matrix(y, n):
    np.set_printoptions(threshold='nan')
    m = np.zeros((len(y), n), dtype=np.int)
    for x in range(len(y)):
        m[x][int(y[x])] = 1   
    return (m)

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
              
    colors = cycle(['red', 'orange', 'yellow', 'green', 'blue',
                    'indigo', 'violet', 'gray', 'black', 'fuchsia',
                    'cadetblue', 'orchid', 'seagreen', 'olive', 'darkgoldenrod',
                    'tomato', 'sienna', 'lightskyblue', 'peru', 'sandybrown'])
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
    
    ## Generifies the script by basing the confusion matrix size off of
    ## the number of gestures in the set in the file. 
    matrixSize = len(set(classifications))
    confMat = np.zeros((matrixSize, matrixSize), dtype=np.int)   
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
    

#    
#    # Classification Report returns a string that contains the chart, but
#    # each element is not in a separate cell. Therefore, classRep splits

#    # the string into an array of each of the elements, with 2 spaces as
#    # the delimiter. 
    classReport = classification_report(allClassifications,allPredictions)
    classRep = classReport.split("  ")
    classRep[3] += ',' #lines up the Precision, Recall, f1, support row with
                       #the rest of the table
    lineUp = len(classRep) - 13 
    classRep[lineUp] += ',' #lines up the avg/total row
            
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
        
################################################################################################################        
    ## needed only for the rawData128SinglePoint.csv file. This function can be 
    ## commented out/removed once new data is collected and used with this script
    ## So when score and test are passed into printRocCurve, they just need the 
    ## matrix function
    allPredictions2 = relabel.replace(allPredictions)
    allClassifications2 = relabel.replace(allClassifications)

    score = matrix(allClassifications2, len(set(allClassifications2)))
    test = matrix(allPredictions2, len(set(allPredictions2)))
################################################################################################################
    
    printRocCurve(f, score, test, isPlot)
    