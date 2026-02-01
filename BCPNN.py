import numpy as np

class BCPNN:
    def __init__(self, hypercolumns, minicolumns, g_beta = 1.0, beta = 1.0, tau_m = 0.02, 
                 g_I = 1.0, g_a = 97.0, tau_p = 10.0, tau_z_pre = 0.15, tau_z_post = 0.005, tau_a=2.70):
        '''A model of the BCPNN network.'''

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
        self.p_pre  = np.ones(self.n_units) / self.n_units
        self.p_post = np.ones(self.n_units) / self.n_units
        self.p_co   = np.ones((self.n_units,self.n_units)) / (self.n_units**2)
        self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        # Variable histories
        self.s_history = []
        self.o_history = [] 
        self.w_ij_history = []      


