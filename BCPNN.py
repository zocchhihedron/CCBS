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
        self.o = np.ones(self.n_units)
        self.a = np.zeros_like(self.o) 
        self.z_pre = np.zeros(self.n_units) * 1.0 / self.minicolumns
        self.z_post = np.zeros(self.n_units) * 1.0 / self.minicolumns

        # Weights and probabilities
        self.w = np.zeros((self.n_units, self.n_units))
        self.p_pre = np.ones(self.n_units) * (1.0 / self.minicolumns)
        self.p_post = np.ones(self.n_units) * (1.0 / self.minicolumns)
        self.p_co = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.beta = np.zeros(self.n_units) # Compatible shape for calculation of s, change afterwards ASAP to correct interpretation
        #self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        # Variable histories for plotting 
        self.s_history = []
        self.o_history = []
        self.w0_history = [] # Example index to check functionality
     
    def update_state(self, I, dt = 1.0, noise = 0.0): # External input pattern-wise
        '''Updates state variables.'''
        # Current 
        self.s += (dt / self.tau_m) * ( + self.g_beta * self.beta  # Bias
                                        + self.g_I * np.dot(self.w.T, self.o) + I  # Internal input current
                                        - self.g_a * self.a  # Adaptation
                                        + noise  # This last term is the noise
                                        - self.s)  # s follow all of the s above  
        self.s_history.append(self.s.copy())

        # WTA mechanism
        argmax = np.argmax(self.s)
        self.o[argmax] = 1.0
        self.o_history.append(self.o.copy())
        print(self.o)


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
        '''Updates weights and bias for training.'''
        print('beta: ', self.beta)

        # Weights  
        eps = 1e-9 # Prevent log(0) as output
        self.w = np.log((self.p_co + eps) / (np.outer(self.p_pre, self.p_post)))

        # Bias
        # self.beta = np.log(self.p_post) FIX!

    def produce_sequences(self, n_patterns = None, s=0, r=0):
        pass

    def train(self, I, pattern_dur, epochs, dt = 1.0):
        '''Trains the network with a sequence as external input.'''
        for epoch in range(epochs): 
            for pattern in seq: # Choose a pattern / time state
                    # One-hot encode the pattern here before updating to ensure the shape matches s and o
                    # Modularize later on
                    encoded_pattern = []
                    for hypercolumn in pattern:
                        hyp_onehot = np.zeros(minicolumns) # one-hot encoding of a single hypercolumn
                        hyp_onehot[hypercolumn] = 1 # the active unit
                        encoded_pattern.append(hyp_onehot) # append each encoding to the encoding of the whole pattern
                        print(encoded_pattern)
                    self.update_state(I = encoded_pattern)
                    self.update_weights()

    def recall(self):
        '''Sequence recall given a cue.'''

        pass

if __name__ == '__main__':
    seq = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    hypercolumns, minicolumns = 3, 3
    nn = BCPNN(hypercolumns, minicolumns)
    nn.train(I = seq, pattern_dur = 1, epochs = 10)

    #plt.plot(nn.s_history_history,nn.o_history)
    #plt.show
