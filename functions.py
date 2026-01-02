import numpy as np
from BCPNN import BCPNN

dt = 0.01

hypercolumns, minicolumns = 5, 3
nn = BCPNN(hypercolumns, minicolumns)

I = np.ones(nn.minicolumns)

def update_state(nn, g_I = nn.g_I, noise = 0, dt = dt):
    '''Updates state variables per time unit without learning.'''

    # Current
    nn.s += (dt / nn.tau_m) * ( + nn.g_beta * nn.beta  # Bias
                                    + nn.g_I * np.dot(nn.w.T, nn.o) + I  # Input current
                                    - nn.g_a * nn.a  # Adaptation
                                    + noise  # Noise
                                    - nn.s)  # s follow all of the s above      







def train_pattern(P, nn, Ndt, I_amp):
    pass

print(nn.beta)