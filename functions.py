import numpy as np
import random
import matplotlib.pyplot as plt
from BCPNN import BCPNN


def update_state(nn, dt, I, noise = 0):
    '''Updates state variables per time unit without learning.'''

    # Current
    nn.s += (dt / nn.tau_m) * ( + nn.g_beta * nn.beta  # Bias
                                    + nn.g_I * I + np.dot(nn.w.T, nn.o)  # Input current
                                    - nn.g_a * nn.a  # Adaptation
                                    + noise  # Noise
                                    - nn.s)  # Current   

    # Unit activations 
    nn.o = strict_max(nn.s, nn.minicolumns)

    # Adaptation
    nn.a += (dt / nn.tau_a) * (nn.o - nn.a)

    # Z-traces   
    nn.z_pre += (dt / nn.tau_z_pre) * (nn.o - nn.z_pre)
    nn.z_post += (dt / nn.tau_z_post) * (nn.o - nn.z_post)

    # Probabilities
    nn.p_pre += (dt / nn.tau_p) * (nn.z_pre - nn.p_pre)
    nn.p_post += (dt / nn.tau_p) * (nn.z_post - nn.p_post)
    nn.p_co += (dt / nn.tau_p) * (np.outer(nn.z_pre, nn.z_post) - nn.p_co)

def update_weights(nn, dt, noise = 0):
    '''Updates weights and biases per time unit for network training.'''

    # Weights  

    eps = 1e-9 # Prevent log(0) as output
    nn.w = np.log((nn.p_co + eps) / (np.outer(nn.p_pre, nn.p_post) + eps))

    # Bias
    nn.beta = np.log(nn.p_post + eps) 

def strict_max(x, minicolumns):
    '''Reshapes the current vector into the unit activation vector.'''

    x = np.reshape(x, (x.size // minicolumns, minicolumns))
    z = np.zeros_like(x)
    maxes = np.argmax(x, axis=1)
    for max_index, max_aux in enumerate(maxes):
        z[max_index, max_aux] = 1

    return z.reshape(x.size)

def train_pattern(nn, dt, Ndt, I, learning = True, save_history = True):
    '''Trains the network on a pattern.'''

    if I is None:
        I = np.zeros(nn.n_units)
    for i in range(Ndt):
        update_state(nn, dt = dt, I = I, noise = 0)
        if learning:
            update_weights(nn, noise = 0, dt = dt)
        if save_history:
            nn.o_history.append(nn.o.copy())
            nn.s_history.append(nn.s.copy())
            if learning:
                nn.w_01_history.append(nn.w[0, 1])

def train_sequence(nn, dt, Ndt, seq, learning = True, save_history = True):
    '''Trains the network on a sequence of patterns.'''

    for pattern in seq:
        train_pattern(nn, dt = dt, Ndt = Ndt, I = pattern, learning = True, save_history = True)

def recall(nn, dt, I_cue, no_patterns, cue_steps, recall_steps):
    '''Recalls a sequence learned by the network by updating the network state without updating weights and biases.'''

    nn.o_history = []

    # Cueing 
    for _ in range(cue_steps):
        update_state(nn, I=I_cue, dt = dt)
        
    # Free recall
    for _ in range(recall_steps):
        update_state(nn, dt = dt, I = np.zeros(nn.n_units))
        nn.o_history.append(nn.o.copy())

def one_hot_encode(pattern, hypercolumns, minicolumns):
    '''Reshapes an indexed pattern representation into a one-hot encoded
    hypercolumn representation'''

    x = np.zeros(hypercolumns * minicolumns)
    for hyp, minic in enumerate(pattern):
        x[hyp * minicolumns + minic] = 1

    return x

def create_sequence(n_patterns, hypercolumns, minicolumns):
    '''Creates a randomized sequence'''
    seq = []
    for n in range(0, n_patterns):
        pattern = []
        for index in range(0, hypercolumns):
            pattern.append(random.randint(0,minicolumns-1))
        seq.append(pattern)
    return seq

def reset_state_probabilities():
    pass

def reset_history(nn):
    nn.s_history = []
    nn.o_history = []    

def plot_o(nn):
    o = np.array(nn.o_history)
    plt.figure(figsize=(10,4))
    plt.imshow(o.T, aspect='auto', cmap='Greys')
    plt.xlabel("Time step")
    plt.ylabel("Unit index")
    plt.title("Unit activations o(t)")
    plt.colorbar(label="Activity")
    plt.tight_layout()
    plt.show()

def plot_s(nn):
    s = np.array(nn.s_history)
    plt.figure(figsize=(10,4))
    plt.imshow(s.T, aspect='auto', cmap='viridis')
    plt.xlabel("Time step")
    plt.ylabel("Unit index")
    plt.title("Membrane currents s(t)")
    plt.colorbar(label="Current")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    dt = 0.01
    Ndt = 50
    hypercolumns = 2
    minicolumns = 3
    n_patterns = 3
    nn = BCPNN(hypercolumns, minicolumns)

    reset_history(nn)

    seq = create_sequence(n_patterns, hypercolumns, minicolumns)
    print(seq)
    seq = np.array([one_hot_encode(p, hypercolumns, minicolumns) for p in seq])
    print(seq)

    train_sequence(nn, dt, Ndt, seq)

    plot_o(nn)
    plot_s(nn)

    



