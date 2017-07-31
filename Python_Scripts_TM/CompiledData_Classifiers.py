# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:07:10 2017

@author: morales
"""

# -*- coding: utf-8 -*-
"""
Created Jun 6 2017

@author: Thatyana

Specific instructions on how to use this script are in the README

Note: This script is run from the IDE itself, not from the command line     

**********To change default values for Log Reg and AdaBoost*******************
The default values are listed right before the LogRegression function down
below. It should look like this:
defaultC = 
defaultPenalty = 
defaultTolerance = 

defaultBase = 
defaultEstimators =

"""

import csv
from numpy import genfromtxt

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron

import CompiledData_PrintChart 


# Default values for Logistic Regression and AdaBoost
defaultC = 10
defaultPenalty = 'l1'
defaultTolerance = .1  

defaultBase = 'dtc'
defaultEstimators = 700

# File IO
originalFile = 'rawData128SinglePoint.csv'
patientID = genfromtxt(originalFile, delimiter=',', skip_header=1,usecols=[384])


# Creates raw data files for each user
for i in set(patientID):
    fileName = 'User%s_RawData.csv' % (int(i))
    userFile = open(fileName, 'wb')
    
    f = open(originalFile, 'rU')
    reader = csv.reader(f)
    writer = csv.writer(userFile)
    for row in reader:
        if int(row[384]) == int(i):
            writer.writerow(row) 
    f.close() 
    userFile.close()
    


# function uses logistic regression classifier 
def logRegression(C, penalty, tolerance, isPlot):
    
     ## Creates CSV file with Classifier type, and parameter values 
     ## (C, Penalty, and Tolerance in that order)
     filetype = 'LogisticRegression_AllUsers_%s_%s_%s.csv' % (str(C), str(penalty), str(tolerance))
     f = open(filetype, "w")
     filename = 'Logistic Regression' + '\nC, ' + str(C) + '\nPenalty, ' + penalty + '\nTolerance, ' + str(tolerance) + '\n\n'
     f.write(filename)    
    
     ## Initializes classifier type, since all 3 of 
     ## the parameters the function takes are strings
     cValue = int(C)
     tolValue = float(tolerance)
     
     ## Sets classifier
     clf = LogisticRegression(C=cValue,penalty=penalty, tol=tolValue)
      
     CompiledData_PrintChart.printChart(f, clf, isPlot)     
            
     ## Statement prints when the file is completed, allowing the user to 
     ## know when to open the file to view it        
     print("\nData is now in \"LogisticRegression_AllUsers_%s_%s_%s.csv\"\n" % (str(C), str(penalty), str(tolerance)))       
                

# function uses support vector machine with a linear kernel
def svMachineLinear(isPlot):
    
    ## Creates CSV file and make title of classifier used
    filetype = 'SVMLinear_AllUsers.csv'
    f = open(filetype, "w")
    f.write("Support Vector Machine: Linear Kernel")
    f.write('\n\n')
    
    ## Classifier type for this is SVM, with a linear kernal 
    clf = svm.SVC(kernel='linear')
     
    CompiledData_PrintChart.printChart(f, clf, isPlot)
              
    print("\nData is now in \"SVMLinear_AllUsers.csv\"\n")  
    
    
# function uses support vector machine with a gaussian/rbf kernel
def svMachineRBF(isPlot):
  
    ## Creates CSV file and makes title of classifier used
    filetype = 'SVMRBF_AllUsers.csv'
    f = open(filetype, "w")
    f.write("Support Vector Machine: Gaussian/RBF Kernel")
    f.write('\n\n')
    
    ## Classifier type for this is SVM, with a gaussian kernal 
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
        
    CompiledData_PrintChart.printChart(f, clf, isPlot)        
      
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"SVMRBF_AllUsers.csv\"\n")    
    
# function uses Decision Tree classifier
def decTree(isPlot):
    
    ## Create CSV file and make fancy title
    filetype = 'DecisionTree_AllUsers.csv'
    f = open(filetype, "w")
    f.write("Decision Tree")
    f.write('\n\n')
    
    ## Classifier type for this is Decision Tree (note, it's considerably faster 
    ## than the other classifiers, in that the results are written sooner)
    clf = tree.DecisionTreeClassifier()
    
    CompiledData_PrintChart.printChart(f, clf, isPlot)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it        
    print("\nData is now in \"DecisionTree_AllUsers.csv\"\n")     


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
    filetype = 'AdaBoost_AllUsers_%s_%s.csv' % (baseEst, nEst)
    f = open(filetype, "w")
    filename = ('AdaBoost with %s\nEstimators:, %s' % (baseEst, nEst))
    f.write(filename)
    f.write('\n\n')
    
    CompiledData_PrintChart.printChart(f, clf, isPlot)
    
    ## Statement prints when the file is completed, allowing the user to know when to open the file to view it   
    filer = '\nData is now in \"AdaBoost_AllUsers_%s_%s.csv\"' % (baseEst, nEstimators)
    print(filer)    
        

def main():    
    
    ## Dictionary to hold classifier function
    classifierDict = {'1':logRegression, '2':svMachineLinear, '3':svMachineRBF, '4':decTree, '5':adaBoost}
    
    print '--logr (default/C=, Penalty=, Tolerance=)\n--svml\n--svmrbf\n--decTree'
    print '--ada (default/base=dtc/rfc/perc, estimators=)\n--all'
    print '\nNote: \'plot\' can be added at the end of any of the classifiers for a pyplot (except all)'
    print 'Which one would you like?'
    clfSelect = str(raw_input())
    choices = clfSelect.split()
            
    # logistic regression
    if choices[0] == '--logr':
        if len(choices) == 1: #default values, no plot
            classifierDict['1'](defaultC, defaultPenalty, defaultTolerance, 'false')
        elif len(choices) == 2:
            if choices[1] == 'plot': #default values with plot
                classifierDict['1'](defaultC, defaultPenalty, defaultTolerance, 'true')
        else:
            if len(choices) == 4:
                classifierDict['1'](choices[1], choices[2], choices[3], 'false') #custom values no plot
            else:
                classifierDict['1'](choices[1], choices[2], choices[3], 'true') #custom values with plot
    
    # SVM linear              
    elif choices[0] == '--svml':
        if len(choices) == 1:
            classifierDict['2']('false') #svm linear no plot
        else:
            if choices[1] == 'plot': #svm linear with plot
                classifierDict['2']('true')
       
    # SVM RBF    
    elif choices[0] == '--svmrbf':
        if len(choices) == 1:
            classifierDict['3']('false') #svm gaussian no plot
        else:
            if choices[1] == 'plot': #svm gaussian with plot
                classifierDict['3']('true')    
    
    # Decision Tree        
    elif choices[0] == '--decTree':
        if len(choices) == 1:
            classifierDict['4']('false') #Decision tree no plot
        else:
            if choices[1] == 'plot': #Decision tree with plot
                classifierDict['4']('true') 
      
    # AdaBoost              
    elif choices[0] == '--ada':
        if len(choices) == 1:
            classifierDict['5'](defaultBase, defaultEstimators, 'false') #default values no plot
        elif len(choices) == 2:
            if choices[1] == 'plot':
                classifierDict['5'](defaultBase, defaultEstimators, 'true') #default values with plot
        elif len(choices) == 3:
            classifierDict['5'](choices[1], int(choices[2]), 'false') #custom values no plot
        else:
            if choices[3] == 'plot':
                classifierDict['5'](choices[1], int(choices[2]), 'true') #custom values with plot
                        
    elif clfSelect == '--all': #this option will just run every classifier with default values and no plot
        classifierDict['1'](defaultC, defaultPenalty, defaultTolerance, 'false')
        classifierDict['2']('false')
        classifierDict['3']('false')
        classifierDict['4']('false')
        classifierDict['5'](defaultBase, defaultEstimators, 'false')
        classifierDict['5']('rfc', defaultEstimators, 'false')
        classifierDict['5']('perc', defaultEstimators, 'false')
                                                      
    else:
        exit(0)  #any key to exit    

if __name__ == "__main__":
    main()
   