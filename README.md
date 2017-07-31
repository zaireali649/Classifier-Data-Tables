##############################################################################
#                                    Overview                                #
##############################################################################
For the Classifier_Algorithms_AllUsers_Command script OR the 
Classifier_Algorithms_OneUser_Command, the user can type this into the command 
line:
    
    python Classifier_Algorithms_AllUsers_Command.py --<classifier_type> <any additional parameters> <plot>

These scripts support running the following classifiers:

    Logistic Regression (Parameters: C, penalty, tolerance)
    SVM Linear
    SVM RBF
    Decision Tree
    AdaBoost (Parameters: base_estimator, n_estimators)
    
A user must enter the script name followed by a 
classifier of their choice:

    Logistic Regression: --logr
    SVM Linear: --svml
    SVM RBF: --svmrbf
    Decision Tree: --dectree
    AdaBoost: --ada 
    
If the user is using the OneUser script, the specified user # can be changed 
within the script. If the user is using the OneUser_NoCommand script, it should 
be run directly from the IDE. If using the AllUsers script, no changes need to 
be made. 

##############################################################################
#                   Additional Parameters and Default Values                 #
##############################################################################
Currently, each classifier can have additional optional parameters. Solely 
typing the classifier name will produce results given by default values

    Logistic Regression has 3 additional parameters:
        C has a default value of 10
        Penalty has a default value of 'l1'
        Tolerance has a default value of 0.1
        
    AdaBoost has 2 additional parameters: **
        base_estimator has a default value of 'dtc' which is Decision Tree
        n_estimators has a default value of 700
        
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

    Default Values without ROC Curve plot
    Default Values with ROC Curve plot
    Custom Values without ROC Curve plot
    Custom Values with ROC Curve plot
    
##############################################################################
#                          CompiledData_Classifiers                          #
##############################################################################
The CompiledData_Classifiers script has only one difference from the other 
Classifier scripts. When run (only from the IDE), it instead outputs one CSV 
file with each user along with their averaged accuracy, F1-score, recall, and 
precision values resulting from the selected classifier. This script is used to 
compile scores for each user so graphs can be easily made to compare classifiers. 
The other scripts are used to output confusion matrices and other tables for 
either all users combined or individual users. This script is better suited to 
prepare data for poster boards and research papers than the others.


##############################################################################
#                                   Examples                                 #
##############################################################################   

The following are acceptable examples of input for each file. Note: only the 
scripts that start with "Classifier_Algorithms" or "CompiledData_Classifiers" 
are to be run The PrintDataTables and PrintChart are accompanying scripts that 
cannot be run independently. 

***Classifier_Algorithms_AllUsers_Command OR Classifier_Algorithms_OneUser_Command***
    C:\Users\[user]\Desktop\[folder]>python Classifier_Algorithms_AllUsers_Command.py --logr
    C:\Users\[user]\Desktop\[folder]>python Classifier_Algorithms_AllUsers_Command.py --ada rfc 500
    C:\Users\[user]\Desktop\[folder]>python Classifier_Algorithms_OneUser_Command.py --svml plot


***Classifier_Algorithms_OneUser_NoCommand OR CompiledData_Classifiers***
These two must be run from the IDE (not the command line). When it runs, the user 
is prompted with the following:
    --logr (default/C=, Penalty=, Tolerance=)
    --svml
    --svmrbf
    --decTree
    --ada (default/base=dtc/rfc/perc, estimators=)
    --all

    Note: 'plot' can be added at the end of any of the classifiers for a pyplot (except all)
    Which one would you like?

The user may proceed as normal with the same format as described with the Command 
files. Only the classifier (and possible additional parameters) need to be entered. 