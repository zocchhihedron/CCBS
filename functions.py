import numpy as np
from BCPNN import BCPNN

dt = 0.01
hypercolumns, minicolumns = 3, 5
nn = BCPNN(hypercolumns, minicolumns)
I = np.ones(nn.minicolumns * nn.hypercolumns)

def update_state(nn, I, g_I = nn.g_I, noise = 0, dt = dt):
    '''Updates state variables per time unit without learning.'''

    # Current
    nn.s += (dt / nn.tau_m) * ( + nn.g_beta * nn.beta  # Bias
                                    + nn.g_I * np.dot(nn.w.T, nn.o) + I  # Input current
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

def update_weights(nn, noise = 0, dt = dt):
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

def train_pattern(nn, Ndt, I, I_amp = nn.g_I, learning = True, save_history = True):
    '''Trains the network on a pattern.'''
    if I.all() == None:
        I = np.zeros(nn.hypercolumns*nn.minicolumns)
    for i in range(Ndt):
        update_state(nn, I, g_I = nn.g_I, noise = 0, dt = dt)
        if learning:
            update_weights(nn, noise = 0, dt = dt)
        if save_history:
            nn.s_history.append(nn.s)
            nn.o_history.append(nn.o)
            if learning:
                nn.w0_history.append(nn.w[0])

def train_sequence(nn, Ndt, seq, I_amp = nn.g_I, learning = True, save_history = True):
    '''Trains the network on a sequence of patterns.'''

    for pattern in seq:
        train_pattern(nn, Ndt, pattern, I_amp = nn.g_I, learning = True, save_history = True)

def recall(nn, I_cue, no_patterns, cue_steps, recall_steps):
    '''Recalls a sequence learned by the network by updating the network state without updating weights and biases.'''
    nn.o_history = []
    # Cueing with first element of the input
    I_zero = np.zeros(nn.n_units)
    for _ in range(no_patterns):
        for _ in range(cue_steps):
            update_state(nn=nn, I=I_cue)
        # The recalled patterns are saved with each step of recall
        nn.o_history.append(nn.o)
        print(nn.o)
    # The history of the unit activations is the recalled sequence
    #print(nn.o_history)

def pattern(indices, hypercolumns, minicolumns):
    '''Reshapes an indexed pattern representation into a one-hot encoded
    hypercolumn representation'''
    x = np.zeros(hypercolumns * minicolumns)
    for hyp, minic in enumerate(indices):
        x[hyp * minicolumns + minic] = 1
    return x

if __name__ == '__main__':
    dt = 0.01
    hypercolumns, minicolumns = 3, 5
    nn = BCPNN(hypercolumns, minicolumns)
    seq = np.array([pattern([0,1,2], 3, 5), pattern([1,1,2], 3, 5)])
    no_patterns = seq.shape[0]
    train_sequence(nn = nn, Ndt = 10, seq = seq, I_amp = nn.g_I, learning = True, save_history = True)
    recall(nn, I_cue = np.zeros(15), cue_steps = 5, no_patterns = no_patterns, recall_steps = 10)




