"""
Module containing all utility needed for full-cycle sequence learning of a BCPNN. 

Includes:
- Sequence generating functions: create_sequence() and one_hot_encode()
- Network-learning functions: update_state() and update_weights() (as well as a helper function strict_max())
- Training-to-recall workflow functions: train_pattern(), train_sequence(), recall() and pause()
- Network reset function: reset_state_probabilities()
- Variable history-documenting functions: clean_history() and update_history()
- Main function for executing the entire sequence learning workflow, containing adjustable parameter 
  values and imported plotting functions for analysis.
"""


import random
import numpy as np
import matplotlib.pyplot as plt
from BCPNN import *
from plot_functions import *


## Sequence generating functions
def create_sequence(n_patterns, hypercolumns, minicolumns):
    '''Creates a randomized sequence'''
    seq = []
    for n in range(0, n_patterns):
        pattern = []
        for index in range(0, hypercolumns):
            pattern.append(random.randint(0,minicolumns-1))
        seq.append(pattern)
    return seq

def one_hot_encode(pattern, hypercolumns, minicolumns):
    '''Reshapes an indexed pattern representation into a one-hot encoded
    hypercolumn representation'''

    x = np.zeros(hypercolumns * minicolumns)
    for hyp, minic in enumerate(pattern):
        x[hyp * minicolumns + minic] = 1

    return x


## Network-learning functions
def update_state(nn, dt, I, noise = 0):
    '''Updates state variables per time unit without learning.'''

    # Current
    nn.s += (dt / nn.tau_m) * ( + nn.g_i * (nn.i_nmda + nn.i_ampa) # NMDA and AMPA effects
                                    + nn.g_beta * nn.beta  # Bias
                                    + nn.g_I * I # Input current
                                    - nn.g_a * nn.a  # Adaptation
                                    + noise  # Noise
                                    - nn.s)  # Current   

    # Unit activations 
    nn.o = strict_max(nn.s, nn.minicolumns)

    # Adaptation
    nn.a += (dt / nn.tau_a) * (nn.o - nn.a)

    # NMDA and AMPA currents (@ indicates matrix multiplication)
    nn.i_nmda = nn.w_nmda @ nn.z_pre_nmda / nn.hypercolumns
    nn.i_ampa = nn.w_ampa @ nn.z_pre_ampa / nn.hypercolumns

    # Z-traces   
    nn.z_pre_nmda += (dt / nn.tau_z_pre_nmda) * (nn.o - nn.z_pre_nmda)
    nn.z_post_nmda += (dt / nn.tau_z_post_nmda) * (nn.o - nn.z_post_nmda)

    nn.z_pre_ampa += (dt / nn.tau_z_pre_ampa) * (nn.o - nn.z_pre_ampa)
    nn.z_post_ampa += (dt / nn.tau_z_post_ampa) * (nn.o - nn.z_post_ampa)

    # Probabilities
    nn.p_pre_nmda += (dt / nn.tau_p_nmda) * (nn.z_pre_nmda - nn.p_pre_nmda)
    nn.p_post_nmda += (dt / nn.tau_p_nmda) * (nn.z_post_nmda - nn.p_post_nmda)
    nn.p_co_nmda += (dt / nn.tau_p_nmda) * (np.outer(nn.z_pre_nmda, nn.z_post_nmda) - nn.p_co_nmda)

    nn.p_pre_ampa += (dt / nn.tau_p_ampa) * (nn.z_pre_ampa - nn.p_pre_ampa)
    nn.p_post_ampa += (dt / nn.tau_p_ampa) * (nn.z_post_ampa - nn.p_post_ampa)
    nn.p_co_ampa += (dt / nn.tau_p_ampa) * (np.outer(nn.z_pre_ampa, nn.z_post_ampa) - nn.p_co_ampa)

def update_weights(nn, dt, noise = 0):
    '''Updates weights and biases per time unit for network training.'''

    # Weights
    eps = 1e-9 # Prevent log(0) 
    nn.w_nmda = np.log((nn.p_co_nmda + eps) / (np.outer(nn.p_pre_nmda, nn.p_post_nmda) + eps))
    nn.w_ampa = np.log((nn.p_co_ampa + eps) / (np.outer(nn.p_pre_ampa, nn.p_post_ampa) + eps))

    # Bias
    nn.beta = np.log(nn.p_post_nmda + eps) 

def strict_max(x, minicolumns):
    '''Helper function to update_state for .'''

    x = np.reshape(x, (x.size // minicolumns, minicolumns))
    z = np.zeros_like(x)
    maxes = np.argmax(x, axis=1)
    for max_index, max_aux in enumerate(maxes):
        z[max_index, max_aux] = 1

    return z.reshape(x.size)


## Training-to-recall workflow functions
def train_pattern(nn, dt, Ndt, I, learning = True, update = True):
    '''Trains the network on a pattern.'''

    if I is None:
        I = np.zeros(nn.n_units)
    for i in range(Ndt):
        update_state(nn, dt = dt, I = I, noise = 0)
        if learning:
            update_weights(nn, noise = 0, dt = dt)
        if update:
            update_history(nn, dt)

def train_sequence(nn, dt, Ndt, seq, learning = True, update = True, IPI=0): 
    '''Trains the network on a sequence of patterns.'''

    for pattern in seq:
        train_pattern(nn, dt = dt, Ndt = Ndt, I = pattern, learning = True, update = True)
        pause(nn, dt, pause_steps = IPI) # = IPI here

def recall(nn, dt, I_cue, cue_steps, recall_steps):
    '''Recalls a sequence learned by the network by updating the network state without updating weights and biases.'''

    # Cueing 
    for _ in range(cue_steps):
        update_state(nn, I=I_cue, dt = dt)
        update_history(nn, dt)
    # Add partial cue
        
    # Cueing recall
    for _ in range(recall_steps):
        update_state(nn, dt = dt, I = np.zeros(nn.n_units)) 
        update_history(nn, dt)

def pause(nn, dt, pause_steps):
    for i in range(pause_steps):
        update_state(nn, dt, I=np.zeros(nn.n_units), noise = 0)
        update_history(nn, dt)


## Network reset function
def reset_state_probabilities(nn):
    nn.w_nmda = np.zeros((nn.n_units, nn.n_units))
    nn.w_ampa = np.zeros((nn.n_units, nn.n_units))
    nn.beta = np.log(np.ones_like(nn.o) * (1.0 / nn.minicolumns))
    nn.p_pre_nmda  = np.ones(nn.n_units) / nn.n_units
    nn.p_post_nmda = np.ones(nn.n_units) / nn.n_units
    nn.p_co_nmda   = np.ones((nn.n_units,nn.n_units)) / (nn.n_units**2)
    nn.p_pre_ampa  = np.ones(nn.n_units) / nn.n_units
    nn.p_post_ampa = np.ones(nn.n_units) / nn.n_units
    nn.p_co_ampa   = np.ones((nn.n_units,nn.n_units)) / (nn.n_units**2)


## Variable history-documenting functions
def clean_history(nn):
    nn.s_history = []
    nn.o_history = []  
    nn.w_01_history = []
    nn.z_pre_nmda_history = []
    nn.z_post_nmda_history = []
    nn.p_pre_nmda_history = []
    nn.p_post_nmda_history = []
    nn.p_co_nmda_history = []
    nn.z_pre_ampa_history = []
    nn.z_post_ampa_history = []
    nn.p_pre_ampa_history = []
    nn.p_post_ampa_history = []
    nn.p_co_ampa_history = []

def update_history(nn, dt):
    nn.s_history.append(nn.s.copy()) 
    nn.o_history.append(nn.o.copy())  
    nn.w_01_history.append(nn.w_nmda[0][1].copy())  
    nn.time_axis.append(nn.time) 
    nn.time += dt
    nn.z_pre_nmda_history.append(nn.z_pre_nmda.copy()) 
    nn.z_post_nmda_history.append(nn.z_post_nmda.copy()) 
    nn.p_pre_nmda_history.append(nn.p_pre_nmda.copy()) 
    nn.p_post_nmda_history.append(nn.p_post_nmda.copy()) 
    nn.p_co_nmda_history.append(nn.p_co_nmda.copy()) 
    nn.z_pre_ampa_history.append(nn.z_pre_ampa.copy()) 
    nn.z_post_ampa_history.append(nn.z_post_ampa.copy()) 
    nn.p_pre_ampa_history.append(nn.p_pre_ampa.copy()) 
    nn.p_post_ampa_history.append(nn.p_post_ampa.copy()) 
    nn.p_co_ampa_history.append(nn.p_co_ampa.copy()) 


## Main function for executing the entire sequence learning workflow
if __name__ == '__main__':

    # Parameter values
    dt = 0.001
    Ndt = 500
    hypercolumns = 3
    minicolumns = 5
    n_patterns = 3
    nn = BCPNN(hypercolumns, minicolumns)
    cue_steps = 100
    recall_steps = 1000
    IPI = 2

    # Network reset
    clean_history(nn)
    reset_state_probabilities(nn)

    # Sequence creation
    seq = create_sequence(n_patterns, hypercolumns, minicolumns)
    print(seq)
    seq = np.array([one_hot_encode(p, hypercolumns, minicolumns) for p in seq])

    # Training & recall with time measurement
    train_sequence(nn, dt, Ndt, seq, IPI)
    print('After training: ', nn.time_axis[-1])
    pause(nn, dt, pause_steps= 10)

    recall(nn, dt, I_cue = seq[0], cue_steps = cue_steps, recall_steps = recall_steps) 
    recall_start = nn.time_axis[-1]
    print('After recall: ', nn.time_axis[-1])

    # Plotting
    o_array = np.array(nn.o_history)
    s_array = np.array(nn.s_history)
    time_array = np.array(nn.time_axis)
    weight_array = np.array(nn.w_01_history)
    p_co_array = np.array(nn.p_co_nmda_history)

    print(weight_array)

    plt.figure(figsize=(12, 8))
    plot_hypercolumn_activations(nn) 




    



