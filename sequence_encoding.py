'''A sandbox for encoding abstractions for sequences used as input in the BCPNN network over 3 levels of 
hierarchy: individual hypercolumns, individual patterns/states and a whole sequence of patterns over time.'''

import numpy as np

def abs1(minicolumns, i=int):
    '''Creates a one-hot-encoded vector representing a hypercolumn with zeros 
    at every position except for the index of the unit which is activated'''
    hypercolumn_act = np.zeros(minicolumns)
    hypercolumn_act[i] = 1
    return hypercolumn_act

def abs2(hypercolumns):
    return

def abs3(patterns):
    return