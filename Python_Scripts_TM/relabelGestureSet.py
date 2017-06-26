# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:32:08 2017

@author: Thatyana Morales

This function is only needed for the rawData128SinglePoint.csv file. After
future data is collected, this file won't be needed, and its use in the 
Classifier_Algorithms.py file can be removed

"""
 
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