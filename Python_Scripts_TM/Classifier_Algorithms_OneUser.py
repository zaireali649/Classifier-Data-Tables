# -*- coding: utf-8 -*-
"""
Created Jun 6 2017

@author: Thatyana

Specific instructions on how to use this script are in the README     

**********To change default values for Log Reg and AdaBoost*******************
The default values are listed right before the LogRegression function down
below. It should look like this:
defaultC = 
defaultPenalty = 
defaultTolerance = 

defaultBase = 
defaultEstimators =

"""

import sys

from numpy import genfromtxt
from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron

import printDataTables_OneUser
import featureExtraction as featExtr

# File IO
path = 'User0SinglePointData.csv'
userNumber = int(filter(str.isdigit, path))
data = genfromtxt(path, delimiter=',', skip_header=1,usecols=range(0,384))
patientID = genfromtxt(path, delimiter=',', skip_header=1,usecols=[384])
classifications = genfromtxt(path, delimiter=',', skip_header=1,usecols=[385])

# Feature Extraction Script Call
features = featExtr.main(data,classifications)

# Subsetting
features = features[:,:]

# Cross- Validation Schemes
kf = KFold(len(classifications), 10, shuffle=True)

# Default values for Logistic Regression and AdaBoost
defaultC = 10
defaultPenalty = 'l1'
defaultTolerance = .01  

defaultBase = 'dtc'
defaultEstimators = 50


# function uses logistic regression classifier 
def logRegression(C, penalty, tolerance, isPlot):
    
     ## Creates CSV file with Classifier type, and parameter values 
     ## (C, Penalty, and Tolerance in that order)
     filetype = 'User_%s_LogisticRegression_%s_%s_%s.csv' % (str(userNumber), str(C), str(penalty), str(tolerance))
     f = open(filetype, "w")
     filename = 'Logistic Regression\n' + 'User, ' + str(userNumber) + '\nC, ' + str(C) + '\nPenalty, ' + penalty + '\nTolerance, ' + str(tolerance) + '\n\n'
     f.write(filename)    
    
     ## Initializes classifier type, since all 3 of 
     ## the parameters the function takes are strings
     cValue = int(C)
     tolValue = float(tolerance)
     
     ## Sets classifier
     clf = LogisticRegression(C=cValue,penalty=penalty, tol=tolValue)
      
     printDataTables_OneUser.printChart(f, clf, isPlot)     
            
     ## Statement prints when the file is completed, allowing the user to 
     ## know when to open the file to view it        
     print("\nData is now in \"User_%s_LogisticRegression_%s_%s_%s.csv\"\n" % (str(userNumber), str(C), str(penalty), str(tolerance)))       
                

# function uses support vector machine with a linear kernel
def svMachineLinear(isPlot):
    
    ## Creates CSV file and make title of classifier used
    filetype = 'User_%s_SVM_Linear.csv' % (str(userNumber))
    f = open(filetype, "w")
    f.write("Support Vector Machine: Linear Kernel")
    f.write('\nUser, ' + str(userNumber) + '\n\n')
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='linear')
     
    printDataTables_OneUser.printChart(f, clf, isPlot)
              
    print("\nData is now in \"User_%s_SVM_Linear.csv\"\n")  
    
    
# function uses support vector machine with a gaussian/rbf kernel
def svMachineRBF(isPlot):
  
    ## Creates CSV file and makes title of classifier used
    filetype = 'User_%s_SVM_RBF.csv' % (str(userNumber))
    f = open(filetype, "w")
    f.write("Support Vector Machine: Gaussian/RBF Kernel")
    f.write('\nUser, ' + str(userNumber) + '\n\n')
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='rbf')
        
    printDataTables_OneUser.printChart(f, clf, isPlot)        
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"User_%s_SVM_RBF.csv\"\n")    
    
# function uses Decision Tree classifier
def decTree(isPlot):
    
    ## Create CSV file and make fancy title
    filetype = 'User_%s_DecisionTree.csv' % (str(userNumber))
    f = open(filetype, "w")
    f.write("Decision Tree")
    f.write('\nUser, ' + str(userNumber) + '\n\n')
    
    ## Classifier type for this is Decision Tree (note, it's considerably faster 
    ## than the other classifiers, in that the results are written sooner)
    clf = tree.DecisionTreeClassifier()
    
    printDataTables_OneUser.printChart(f, clf, isPlot)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"User_%s_DecisionTree.csv\"\n")     


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
    filetype = 'User_%s_AdaBoost_%s_%s.csv' % (str(userNumber), baseEst, nEst)
    f = open(filetype, "w")
    filename = ('AdaBoost with %s\nEstimators:, %s' % (baseEst, nEst))
    f.write(filename)
    f.write('\nUser, ' + str(userNumber) + '\n\n')
    
    printDataTables_OneUser.printChart(f, clf, isPlot)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it   
    filer = '\nData is now in User_%s_AdaBoost_%s_%s.csv' % (str(userNumber), baseEst, nEstimators)
    print(filer)    


def main():

    clfSelect = sys.argv[1]  
    
    ## Dictionary to hold classifier function
    classifierDict = {'1':logRegression, '2':svMachineLinear, '3':svMachineRBF, '4':decTree, '5':adaBoost}

    # logistic regression
    if clfSelect == '--logr':
        if len(sys.argv) == 2: #default values, no plot
            classifierDict['1'](defaultC, defaultPenalty, defaultTolerance, 'false')
        elif len(sys.argv) == 3:
            if sys.argv[2] == 'plot': #default values with plot
                classifierDict['1'](defaultC, defaultPenalty, defaultTolerance, 'true')
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
            classifierDict['5'](defaultBase, defaultEstimators, 'false') #default values no plot
        elif len(sys.argv) == 3:
            if sys.argv[2] == 'plot':
                classifierDict['5'](defaultBase, defaultEstimators, 'true') #default values with plot
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