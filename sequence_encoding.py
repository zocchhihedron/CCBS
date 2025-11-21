'''A sandbox for encoding abstractions for sequences used as input in the BCPNN network over 3 levels of 
hierarchy: individual hypercolumns, individual patterns/states and a whole sequence of patterns over time.'''

import numpy as np

def hypercolumn_state(minicolumns, i=int):
    '''Creates vectors representing hypercolumns with zeros at every 
    position except for the ith element of the unit, which is activated.'''
    hypercolumn_state = np.zeros(minicolumns)
    hypercolumn_state[i] = 1
    activation_index = i
    return hypercolumn_state, activation_index

def pattern(hypercolumns, hypercolumn_states):
    '''Creates a vector representing a pattern with the value of nth index 
    representing the index of the unit being activated in the nth hypercolumn.'''
    pattern = np.zeros(hypercolumns)
    i = 0
    for n in hypercolumn_states:
        pattern[n] = hypercolumn_states[n]
    return pattern

def sequence(n_patterns, minicolumns, patterns):
    '''Creates an array representing a sequence of patterns over time with the
    ith row representing the ith pattern/time state and its jth column representing
    the jth hypercolumn in the pattern.'''
    seq = np.empty((n_patterns, minicolumns))
    for i in seq:
        for j in seq:
            seq[i][j] = patterns[i][j]
    return seq

hypercolumn_states_0 = [hypercolumn_state(3,1)[1], hypercolumn_state(3,0)[1], hypercolumn_state(3,0)[1]]
hypercolumn_states_1 = [hypercolumn_state(3,0)[1], hypercolumn_state(3,1)[1], hypercolumn_state(3,0)[1]]
hypercolumn_states_2 = [hypercolumn_state(3,0)[1], hypercolumn_state(3,0)[1], hypercolumn_state(3,2)[1]]
print(hypercolumn_states_0)
pattern_0 = pattern(3, hypercolumn_states_0)
pattern_1 = pattern(3, hypercolumn_states_1)
pattern_2 = pattern(3, hypercolumn_states_2)
print(pattern_0)
patterns = [pattern_0, pattern_1, pattern_2]
print(patterns)
seq = sequence(3, 3, patterns)
print(seq)