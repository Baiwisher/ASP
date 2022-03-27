# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        

    
    def st_mc(self, spot, texp=None, sigma=None, N_intervals = 100):
        t_delte = texp / N_intervals
        sigma_sequence = []
        sigma_sequence.append(self.sigma)
        s_sequence_log = []
        s_sequence_log.append(np.log(spot))


        for i in range(0,N_intervals):
            a = np.random.normal(size=2)
            W_1 = a[0]
            Z_1 = self.rho * a[0] + np.sqrt(1 - self.rho**2) * a[1]
            sigma_sequence.append(sigma_sequence[i]*np.exp(self.vov * np.sqrt(t_delte) \
                                  * Z_1 - 0.5 * self.vov ** 2 * t_delte))
            s_sequence_log.append(s_sequence_log[i] + sigma_sequence[i] * np.sqrt(t_delte) \
                                  * W_1 - 0.5 * sigma_sequence[i] ** 2 * t_delte)

        sigma_sequence = np.array(sigma_sequence)
        s_sequence = np.exp(np.array(s_sequence_log))

        return s_sequence[len(s_sequence) - 1]
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        sample_population = 100000
        price_matrix = np.zeros([sample_population,len(strike)])
        sigma = sigma if(sigma != None) else self.sigma
        for i in range(0,sample_population):
            st = np.array([self.st_mc(spot, texp, sigma)] \
                          * len(strike))
            st_strike_delta = np.append([st - strike],[np.zeros(len(strike))], axis = 0)
            st_strike = st_strike_delta.max(0)
            price_matrix[i] = st_strike    
#         np.random.seed(12345)
        return price_matrix.sum(0) * 1/sample_population

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def st_mc(self, spot, texp=None, sigma=None, N_intervals = 100):
        t_delte = texp / N_intervals
        sigma_sequence = []
        sigma_sequence.append(self.sigma)
        s_sequence = []
        s_sequence.append(spot)


        for i in range(0,N_intervals):
            a = np.random.normal(size=2)
            W_1 = a[0]
            Z_1 = self.rho * a[0] + np.sqrt(1 - self.rho**2) * a[1]
            sigma_sequence.append(sigma_sequence[i]*np.exp(self.vov * np.sqrt(t_delte) \
                                  * Z_1 - 0.5 * self.vov ** 2 * t_delte))
            s_sequence.append(s_sequence[i] + sigma_sequence[i] * np.sqrt(t_delte) * W_1)

        sigma_sequence = np.array(sigma_sequence)
        s_sequence = np.array(s_sequence)

        return s_sequence[len(s_sequence) - 1]
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        sample_population = 100000
        price_matrix = np.zeros([sample_population,len(strike)])
        sigma = sigma if(sigma != None) else self.sigma
        for i in range(0,sample_population):
            st = np.array([self.st_mc(spot, texp, sigma)] \
                          * len(strike))
            st_strike_delta = np.append([st - strike],[np.zeros(len(strike))], axis = 0)
            st_strike = st_strike_delta.max(0)
            price_matrix[i] = st_strike    
#         np.random.seed(12345)
        return price_matrix.sum(0) * 1/sample_population

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, spot, texp, sigma):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        
        '''
        np.random.seed(123456)
        a = np.random.normal(size=[2,100])
        Z_1 = a[0]
        X_1 = rho * a[0] + np.sqrt(1 - rho**2) * a[1]
        
        t_delte = texp / N_intervals
        sigma_sequence = []
        sigma_sequence.append(self.sigma)

        for i in range(0,N_intervals):
            sigma_sequence.append(sigma_sequence[i]*np.exp(self.vov * np.sqrt(t_delte) \
                                  * Z_1 - 0.5 * self.vov ** 2 * t_delte))
        sigma_sequence = np.array(sigma_sequence)
        
        return sigma_sequence
    
    def price(self, strike, spot, texp=None, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        
        np.random.seed(123456)
        a = np.random.normal(size=[2,100])
        Z_1 = a[0]
        X_1 = rho * a[0] + np.sqrt(1 - rho**2) * a[1]
        
        sigma_sequence = self.bsm_vol(spot, texp, 1)
        I_T = (sigma_sequence ** 2).sum() / 100
        sigma_T = 1 * np.exp(self.vov * Z_1[100] - 0.5 * self.vov ** 2 * texp)
        s_0 = spot * exp(self.rho / self.vov * (sigma_T - 1) - 0.5 * \
                        self.rho ** 2 * 1 * texp * I_T)
        sigma_BS = 1 * np.sqrt((1 - self.rho ** 2) * I_T)
        m_bsm = pf.Bsm(sigma_BS)
        price_continuous = m_bsm.price(strike, s_0, texp)
        print("check")
        return price_continuous
        

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        np.random.seed(123456)
        a = np.random.normal(size=[2,100])
        Z_1 = a[0]
        X_1 = rho * a[0] + np.sqrt(1 - rho**2) * a[1]
        
        t_delte = texp / N_intervals
        sigma_sequence = []
        sigma_sequence.append(self.sigma)

        for i in range(0,N_intervals):
            sigma_sequence.append(sigma_sequence[i]*np.exp(self.vov * np.sqrt(t_delte) \
                                  * Z_1 - 0.5 * self.vov ** 2 * t_delte))
        sigma_sequence = np.array(sigma_sequence)
        
        return sigma_sequence
    
    def price(self, strike, spot, cp=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
                np.random.seed(123456)
        a = np.random.normal(size=[2,100])
        Z_1 = a[0]
        X_1 = rho * a[0] + np.sqrt(1 - rho**2) * a[1]
        
        sigma_sequence = self.bsm_vol(spot, texp, 1)
        I_T = (sigma_sequence ** 2).sum() / 100
        sigma_T = 1 * np.exp(self.vov * Z_1[100] - 0.5 * self.vov ** 2 * texp)
        s_0 = spot + self.rho / self.vov * (sigma_T - 1)
        sigma_N = 1 * np.sqrt((1 - self.rho ** 2) * I_T)
        m_norm = pf.Norm(sigma_N)
        price_continuous = m_norm.price(strike, s_0, texp)
        return price_continuous
