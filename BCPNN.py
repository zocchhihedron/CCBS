import numpy as np

class BCPNN:
    def __init__(self, hypercolumns, minicolumns, g_beta = 1.0, beta = 1.0, tau_m = 0.02, 
                 g_I = 10.0, g_a = 97.0, g_i = 1.0, tau_p_nmda = 1.0, tau_p_ampa = 1.0, tau_z_pre_nmda = 0.15, 
                 tau_z_post_nmda = 0.05, tau_z_pre_ampa = 5.0, tau_z_post_ampa = 2.0, tau_a=20):
        '''A model of the BCPNN network.'''

        # Matrix representation
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns
        self.n_units = self.hypercolumns * self.minicolumns

        # Parameters
        self.g_beta = g_beta 
        self.g_I = g_I
        self.g_a = g_a
        self.g_i = g_i
        self.tau_m  = tau_m
        self.tau_p_nmda = tau_p_nmda
        self.tau_p_ampa = tau_p_ampa
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
        self.i_nmda = np.zeros(self.n_units)
        self.i_ampa = np.zeros(self.n_units)

        self.w_nmda = np.zeros((self.n_units, self.n_units))
        self.w_ampa = np.zeros((self.n_units, self.n_units))

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
        self.w_01_history = []       
        self.z_pre_nmda_history = []
        self.z_post_nmda_history = []
        self.p_pre_nmda_history = []
        self.p_post_nmda_history = []
        self.p_co_ampa_history = []  
        self.z_pre_ampa_history = []
        self.z_post_ampa_history = []
        self.p_pre_ampa_history = []
        self.p_post_ampa_history = []
        self.p_co_ampa_history = []  

        self.time = 0.0
        self.time_axis = []


