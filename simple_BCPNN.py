import numpy as np
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
            I = np.ndarray((minicolumns, int(dt)))
            print('I state upd' , I)

        # Current 
        self.s += (dt / self.tau_m) * ( + self.g_beta * self.beta  # Bias
                                        + self.g_I * np.dot(self.w.T, self.o)  # Internal input current
                                        - self.g_a * self.a  # Adaptation
                                        + noise  # This last term is the noise
                                        - self.s)  # s follow all of the s above  
        
        # WTA mechanism
        self.o = self.softmax(self.s)

        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)

        # Z-traces   
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)

        # Probabilities
        self.p_pre += (dt / self.tau_p) * (self.z_pre - self.p_pre)
        self.p_post += (dt / self.tau_p) * (self.z_post - self.p_post)
        self.p_co += (dt / self.tau_p) * (np.outer(self.z_pre, self.z_post))

    def update_weights(self, dt = 1.0, I = None, noise = 0.0):
        '''Updates weights for training.'''
        # External input current
        if I is None:
            I = np.ndarray((minicolumns, int(dt)))
            print('I weight upd' , I)

        # Weights  
        eps = 1e-9 # Prevent log(0) as output
        self.w = np.log((self.p_co + eps) / (np.outer(self.p_pre, self.p_post)))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sequence(self, overlap = 0):
        '''Creates 2 sequences containing patterns with the chosen degree of overlap.'''
        n_shared = int(overlap * self.n_units)
        n_unique = self.n_units - n_shared

        seq1 = np.arange(self.n_units)

        shared_part = seq1[:n_shared] 
        unique_part = np.arange(self.n_units, self.n_units + n_unique) 
        seq2 = np.concatenate([shared_part, unique_part])

        return seq1, seq2

    def train(self, sequence, pattern_dur, epochs, dt = 1.0):
        '''Trains the network.'''
        for epoch in range(epochs):
            for pattern_idx in sequence:
                I = np.ndarray((minicolumns, int(dt)))
                for _ in range(pattern_dur):
                    self.update_state(dt, I)
                    self.update_weights(dt, I)

    def recall(self, cue, steps=10, dt=1.0, noise=0.0):
        '''Sequence recall given a cue.'''
        # Initialize recall state
        self.o = cue.copy()
        recall_history = [self.o.copy()]

        for t in range(steps):
            # Recurrent input from learned weights
            I_recurrent = np.dot(self.w.T, self.o)
            print('I_recurrent', I_recurrent)
            
            # Update network state using the recurrent input
            self.update_state(dt=dt, I=I_recurrent, noise=noise)
            
            # Store output pattern
            recall_history.append(self.o.copy())

        return np.array(recall_history)

if __name__ == '__main__':
    # Usage example
    hypercolumns = 1
    minicolumns = 2
    nn = BCPNN(hypercolumns, minicolumns)
    seq1, seq2 = nn.sequence(overlap = 0)
    nn.train(seq1, pattern_dur = 1, epochs = 10, dt = 0.01)
    nn.update_state()
    nn.update_weights()
    cue = np.eye(nn.n_units)[seq1[0]]
    recall = nn.recall(cue, steps=10, dt=0.01)
    print('Pattern: ', seq1)
    print('Recall: ', recall)
