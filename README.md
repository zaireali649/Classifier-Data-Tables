##############################################################################
#                                    Overview                                #
##############################################################################
For the Classifier Algorithms script, the user can type this into the command line:
    
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