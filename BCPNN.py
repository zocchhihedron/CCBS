"""
A Bayesian Confidence Propagation Neural Network (BCPNN) class based off the previous work 
done at the Computational Brain Science Lab at KTH Royal University of Technology.

Contains all representation, parameters, state variables as well as history-documenting
functions and a built-in time axis for a BCPNN with an arbitrary number of hypercolumns 
and minicolumns. The network encompasses dual-receptor utility with both NMDA and AMPA 
channels. All parameters are scaled by a factor dt in the functions module during 
application.
"""


import numpy as np


class BCPNN:
    def __init__(self, hypercolumns, minicolumns, g_beta = 1.0, beta = 10.0, tau_m = 0.02, 
                 g_I = 100.0, g_a = 97.0, g_i = 10.0, tau_p_nmda = 5.0, tau_p_ampa = 5.0, 
                 tau_z_pre_nmda = 0.15, tau_z_post_nmda = 0.005, tau_z_pre_ampa = 0.005, 
                 tau_z_post_ampa = 0.005, tau_a=2.0, noise=0.0): #tau_p larger for longer sequences
        '''An adjustable model of a Bayesian Confidence Propagation Neural Network (BCPNN).'''


        self.noise = noise
        
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

        # Time axis for analysis
        self.time = 0.0
        self.time_axis = []


