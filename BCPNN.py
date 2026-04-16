import numpy as np

class BCPNN:
    def __init__(self, hypercolumns, minicolumns, g_beta = 1.0, beta = 1.0, tau_m = 0.02, 
                 g_I = 10.0, g_a = 25.0, tau_p = 1.0, tau_z_pre_nmda = 0.1, tau_z_post_nmda = 0.02,
                 tau_z_pre_ampa = 0.1, tau_z_post_ampa = 0.02, tau_a=2.70):
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
        self.tau_z_pre_nmda = tau_z_pre_nmda
        self.tau_z_post_nmda = tau_z_post_nmda
        self.tau_z_pre_ampa = tau_z_pre_ampa
        self.tau_z_post_ampa = tau_z_post_ampa
        self.tau_a = tau_a

        # State variables
        self.s = np.zeros(self.n_units) 
        self.o = np.ones(self.n_units)
        self.a = np.zeros_like(self.o) 
        
        self.z_pre_nmda = np.zeros(self.n_units) * 1.0 / self.minicolumns
        self.z_post_nmda = np.zeros(self.n_units) * 1.0 / self.minicolumns

        self.z_pre_ampa = np.zeros(self.n_units) * 1.0 / self.minicolumns
        self.z_post_ampa = np.zeros(self.n_units) * 1.0 / self.minicolumns

        # Weights and probabilities
        self.w = np.zeros((self.n_units, self.n_units))
        self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        self.p_pre_nmda  = np.ones(self.n_units) / self.n_units
        self.p_post_nmda = np.ones(self.n_units) / self.n_units
        self.p_co_nmda   = np.ones((self.n_units,self.n_units)) / (self.n_units**2)

        self.p_pre_ampa  = np.ones(self.n_units) / self.n_units
        self.p_post_ampa = np.ones(self.n_units) / self.n_units
        self.p_co_ampa  = np.ones((self.n_units,self.n_units)) / (self.n_units**2)


        # Variable histories
        self.s_history = []
        self.o_history = []    
        self.z_pre_history = []
        self.z_post_history = []
        self.p_pre_history = []
        self.p_post_history = []
        self.p_co_history = []
        self.w_01_history = []    

        self.time = 0.0
        self.time_axis = []


