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

    # nmda & ampa distinction

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

def pause(nn, dt, pause_steps):
    for i in range(pause_steps):
        update_state(nn, dt, I=np.zeros(nn.n_units), noise = 0)
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

def reset_state_probabilities(nn):
    nn.w = np.zeros((nn.n_units, nn.n_units))
    nn.p_pre  = np.ones(nn.n_units) / nn.n_units
    nn.p_post = np.ones(nn.n_units) / nn.n_units
    nn.p_co   = np.ones((nn.n_units,nn.n_units)) / (nn.n_units**2)
    nn.beta = np.log(np.ones_like(nn.o) * (1.0 / nn.minicolumns))

def update_history(nn, dt):
    nn.s_history.append(nn.s.copy()) 
    nn.o_history.append(nn.o.copy())  
    nn.z_pre_history.append(nn.z_pre.copy()) 
    nn.z_post_history.append(nn.z_post.copy()) 
    nn.p_pre_history.append(nn.p_pre.copy()) 
    nn.p_post_history.append(nn.p_post.copy()) 
    nn.p_co_history.append(nn.p_co.copy()) 
    nn.w_01_history.append(nn.w[0][1].copy())  
    nn.time_axis.append(nn.time) 
    nn.time += dt

def clean_history(nn):
    nn.s_history = []
    nn.o_history = []  
    nn.z_pre_history = []
    nn.z_post_history = []
    nn.p_pre_history = []
    nn.p_post_history = []
    nn.p_co_history = []
    nn.w_01_history = []

def plot_hypercolumn_activations(nn, gap=1):
    o_array = np.array(nn.o_history)
    time_array = np.array(nn.time_axis)
    
    plt.figure(figsize=(12, 8))
    
    # Track labels for the y-axis
    y_ticks = []
    y_labels = []

    for h in range(nn.hypercolumns):
        for m in range(nn.minicolumns):
            # Calculate global index in the flat array
            unit_idx = h * nn.minicolumns + m
            
            # Calculate shifted y-position with a gap between hypercolumns
            y_pos = (h * (nn.minicolumns + gap)) + m
            
            # Find when this unit was active
            active_indices = np.where(o_array[:, unit_idx] == 1)[0]
            
            if len(active_indices) > 0:
                plt.scatter(time_array[active_indices], 
                            np.ones_like(active_indices) * y_pos, 
                            marker='s', s=40, color='black')
            
            # Record tick position and label
            y_ticks.append(y_pos)
            y_labels.append(f"H{h}:M{m}")

    # Set custom ticks to show Hypercolumn and Minicolumn IDs
    plt.yticks(y_ticks, y_labels, fontsize=8)
    plt.ylabel("Hypercolumn (H) : Minicolumn (M)")
    plt.xlabel("Time (s)")
    plt.title("BCPNN Activation: Grouped by Hypercolumns")
    
    # Optional: Add horizontal lines to separate hypercolumns visually
    for h in range(1, nn.hypercolumns):
        line_pos = h * (nn.minicolumns + gap) - (gap / 2)
        plt.axhline(y=line_pos, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    dt = 0.01
    Ndt = 50
    hypercolumns = 3
    minicolumns = 3
    n_patterns = 3
    nn = BCPNN(hypercolumns, minicolumns)
    cue_steps = 10
    recall_steps = 30
    IPI = 2

    clean_history(nn)
    reset_state_probabilities(nn)

    #seq = create_sequence(n_patterns, hypercolumns, minicolumns)
    seq = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    seq = np.array([one_hot_encode(p, hypercolumns, minicolumns) for p in seq])
    print(seq)
    print(seq[0])

    train_sequence(nn, dt, Ndt, seq, IPI)
    print('After training: ', nn.time_axis[-1])
    pause(nn, dt, pause_steps= 10)
    print('After pause: ', nn.time_axis[-1])
    

    # Choose time axis instead of deleting history
    recall(nn, dt, I_cue = seq[0], cue_steps = cue_steps, recall_steps = recall_steps) 
    # NEXT: Make sure learning happens -> Scale to 10x10 with random sequences and check if they are recalled

    print('After recall: ', nn.time_axis[-1])

    # Convert history lists to numpy arrays
    # o_history shape: (time_steps, n_units)
    o_array = np.array(nn.o_history)
    time_array = np.array(nn.time_axis)
    
    plt.figure(figsize=(12, 8))
    
    plot_hypercolumn_activations(nn, gap=1) 

# Test maximal pattern amount to be stored (bereonde på sekvenslängd) = scaling of the newtork -> random sequences
# Try first sequence with 100 units (10x10)



    



