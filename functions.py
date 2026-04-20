"""
Module containing all utility needed for full-cycle sequence learning of a BCPNN. 

Includes:
- Sequence generating functions: create_sequence() and one_hot_encode()
- Network-learning functions: update_state() and update_weights() (as well as a helper function soft_max())
- Training-to-recall workflow functions (building blocks): train_pattern(), train_sequence(), recall() and pause()
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
    """
    Creates a randomized sequence of N patterns shaped
    H x 1 and with entries between 0 and M - 1.

    Parameters:
        n_patterns: Number of patterns in the sequence (N).
        hypercolumns: Number of hypercolumns per pattern (H).
        minicolumns: Number of minicolumns per hypercolumn (M).

    Returns:
        seq: An array of N (H x 1)-vectors with entries in range
        [0, M-1].
    """
    seq = []
    for n in range(0, n_patterns):
        pattern = []
        for index in range(0, hypercolumns):
            pattern.append(random.randint(0,minicolumns-1))
        seq.append(pattern)
    return seq

def one_hot_encode(pattern, hypercolumns, minicolumns):
    """
    Re-shapes a (H x 1)-vector into a one-hot-encoded vector 
    (indexed pattern representation to one-hot encoded 
    hypercolumn representation).

    Parameters:
        pattern: The (H x 1)-vector.
        hypercolumns: Number of hypercolumns per pattern (H).
        minicolumns: Number of minicolumns per hypercolumn (M).

    Returns:
        ohe_pattern: A one-hot encoded (H * M x 1)-vector divided
        into H sections, where all entries in one section are zero
        except for the index equaling the minicolumn that is active.
    """

    ohe_pattern = np.zeros(hypercolumns * minicolumns)
    for hyp, minic in enumerate(pattern):
        ohe_pattern[hyp * minicolumns + minic] = 1

    return ohe_pattern


## Network-learning functions
def update_state(nn, dt, I, noise = 0):
    """
    Updates state variables in an instance of the BPNN network in accordance
    to theory. The state variables should be updated with every run of the 
    network, whether it is learning or not.

    Parameters:
        nn: An instance of the BCPNN network.
        dt: Time-scaling factor for the network parameters.
        noise: Possible noise term (zero at default).
    """

    # Current
    nn.s += (dt / nn.tau_m) * ( + nn.g_i * (nn.i_nmda + nn.i_ampa) # NMDA and AMPA effects
                                    + nn.g_beta * nn.beta  # Bias
                                    + nn.g_I * I # Input current
                                    - nn.g_a * nn.a  # Adaptation
                                    + noise  # Noise
                                    - nn.s)  # Current   

    # Unit activations 
    nn.o = soft_max(nn.s, nn.minicolumns)

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
    """
    Updates weights and biases in an instance of the BPNN network in accordance
    to theory. This is where sequence learning happens.

    Parameters:
        nn: An instance of the BCPNN network.
        dt: Time-scaling factor for the network parameters.
        noise: Possible noise term (zero at default).
    """

    # Weights
    eps = 1e-9 # Prevent log(0) 
    nn.w_nmda = np.log((nn.p_co_nmda + eps) / (np.outer(nn.p_pre_nmda, nn.p_post_nmda) + eps))
    nn.w_ampa = np.log((nn.p_co_ampa + eps) / (np.outer(nn.p_pre_ampa, nn.p_post_ampa) + eps))

    # Bias (Currently a function of NMDA only)
    nn.beta = np.log(nn.p_post_nmda + eps) 

def soft_max(x, minicolumns):
    """
    A helper function for transforming the current s into normalized 
    unit activations using a Softmax distribution per hypercolumn.

    Parameters:
        x: The input current vector (size: hypercolumns * minicolumns).
        minicolumns: Number of minicolumns per hypercolumn.

    Returns:
        z.flatten(): Normalized activations summing to 1 per hypercolumn.
    """
    # Reshape to (Hypercolumns, Minicolumns)
    x_reshaped = x.reshape(x.size // minicolumns, minicolumns)
    
    # Subtract max for numerical stability (prevents overflow in exp)
    shift_x = x_reshaped - np.max(x_reshaped, axis=1, keepdims=True)
    
    # Apply exponential with temperature scaling
    exp_x = np.exp(shift_x)
    
    # Normalize across the minicolumns axis
    z = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    return z.flatten()


## Training-to-recall workflow functions (building blocks)
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
    """
    Resets all state probabilities in an instance of the BCPNN network.

    Parameters:
        nn: An instance of the BCPNN network.
    """
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
    """
    Resets all variable histories in an instance of the BCPNN network.

    Parameters:
        nn: An instance of the BCPNN network.
    """
    
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
    """
    Updates all variable histories in an instance of the BCPNN network.

    Parameters:
        nn: An instance of the BCPNN network.
    """

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

    print(o_array)

    plt.figure(figsize=(12, 8))
    plot_hypercolumn_activations(nn) 




    



