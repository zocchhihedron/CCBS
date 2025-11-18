import numpy as np
import math
import matplotlib as plt

class BCPNN:
    def __init__(self, hypercolumns, minicolumns, g_beta = 1.0, beta = 1.0, tau_m = 0.02, 
                 g_I = 1.0, g_a = 97.0, tau_p = 10.0, tau_z_pre = 0.15, tau_z_post = 0.005, tau_a=2.70):
        '''A simplified model of the BCPNN network with defined network variables and functions for creating
        sequences, training the network, updating variables and running the network for given sequences.
        A helper softmax function for the updating function is included. Default values for instance variables 
        taken from Table 1 in the paper.'''

        # Matrix representation
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns
        self.n_units = self.hypercolumns * self.minicolumns

        # Parameters
        self.g_beta = g_beta 
        self.g_I = g_I
        self.g_a = g_a
        self.tau_m  = tau_m
        self.tau_p = tau_p
        self.tau_z_pre = tau_z_pre
        self.tau_z_post = tau_z_post
        self.tau_a = tau_a

        # State variables
        self.s = np.zeros(self.n_units)   
        self.o = np.ones(self.n_units) * (1.0 / self.minicolumns) # MODIFY?
        self.a = np.zeros_like(self.o) 
        self.z_pre = np.zeros(self.n_units) * 1.0 / self.minicolumns
        self.z_post = np.zeros(self.n_units) * 1.0 / self.minicolumns

        # Weights and probabilities
        self.w = np.zeros((self.n_units, self.n_units))
        self.p_pre = np.ones(self.n_units) * (1.0 / self.minicolumns)
        self.p_post = np.ones(self.n_units) * (1.0 / self.minicolumns)
        self.p_co = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))
     
    def update_state(self, dt = 1.0, I = None, noise = 0.0):
        '''Updates state variables.'''
        # External input current
        if I is None:
            I = np.ndarray((minicolumns, math.ceil(dt))) #Round dt up to the closest integer (avoid 0)


        # Current 
        self.s += (dt / self.tau_m) * ( + self.g_beta * self.beta  # Bias
                                        + self.g_I * np.dot(self.w.T, self.o)  # Internal input current
                                        - self.g_a * self.a  # Adaptation
                                        + noise  # This last term is the noise
                                        - self.s)  # s follow all of the s above  
        print('s type: ', np.shape(self.s))

        # WTA mechanism
        self.o = np.argmax(self.s)
        print('s is ', self.s)
        print('o is ', self.o)

        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)

        # Z-traces   
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)

        # Probabilities
        self.p_pre += (dt / self.tau_p) * (self.z_pre - self.p_pre)
        self.p_post += (dt / self.tau_p) * (self.z_post - self.p_post)
        self.p_co += (dt / self.tau_p) * (np.outer(self.z_pre, self.z_post))

    def update_weights(self, dt = 5.0, I = None, noise = 0.0):
        '''Updates weights for training.'''
        # External input current
        if I is None:
            I = np.ndarray((minicolumns, math.ceil(dt))) #Round dt up to the closest integer (avoid 0)

        # Weights  
        eps = 1e-9 # Prevent log(0) as output
        self.w = np.log((self.p_co + eps) / (np.outer(self.p_pre, self.p_post)))

    def produce_sequences(self, n_patterns = None, s=0, r=0):
        '''Creates 2 sequences containing patterns with the chosen degree of element-wise and temporal overlap
        (s = sequential overlap, r = representational overlap)'''
        if n_patterns == None:
            n_patterns = int(minicolumns)

        n_r = int(r * n_patterns / 2)
        n_s = int(s * hypercolumns)
        n_size = int(n_patterns / 2)

        # Create orthogonal canonical representation
        aux = []
        for i in range(minicolumns):
            aux.append(i * np.ones(hypercolumns))
        matrix = np.array(aux, dtype = 'int')[:n_patterns]

        seq1 = matrix[:n_size]
        seq2 = matrix[n_size:]

        start_index = max(int(0.5 * (n_size - n_r)), 0)
        end_index = min(start_index + n_r, n_size)

        for index in range(start_index, end_index):
            seq2[index, :n_s] = seq1[index, :n_s]

        return seq1, seq2

    def train(self, sequence, pattern_dur, epochs, dt = 1.0):
        '''Trains the network.'''
        for epoch in range(epochs):
            for pattern in sequence:
                if dt < 1.0:
                    I = np.ndarray((minicolumns, 1))
                else:

                    I = np.ndarray((minicolumns, int(dt)))
                for pattern in range(pattern_dur):
                    self.update_state(dt, I)
                    self.update_weights(dt, I)

    def recall(self, cue, steps=10, dt=1.0, noise=0.0):
        '''Sequence recall given a cue.'''
        # Initialize recall state
        self.o = cue.copy()
        recall_history = [self.o.copy()]

        for i in range(steps):
            # Inner input from learned weights
            I_inner = np.dot(self.w.T, self.o)

            # Update network state using the inner input
            self.update_state(dt=dt, I=I_inner, noise=noise)
            
            # Store output pattern
            recall_history.append(self.o.copy())

        return np.array(recall_history)

if __name__ == '__main__':
    # Usage example
    hypercolumns = 3
    minicolumns = 4
    nn = BCPNN(hypercolumns, minicolumns)
    seq1, seq2 = nn.produce_sequences(r = 1)
    print('seq1: ', seq1)
    print('seq2: ', seq2)   
