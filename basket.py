# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""
import numpy as np
import scipy.stats as ss
import pyfeng as pf

def basket_check_args(spot, vol, corr_m, weights):
    '''
    This function simply checks that the size of the vector (matrix) are consistent
    '''
    n = spot.size
    assert( n == vol.size )
    assert( corr_m.shape == (n, n) )
    return None
    
def basket_price_mc_cv(
    strike, spot, vol, weights, texp, cor_m, 
    intr=0.0, divr=0.0, cp=1, n_samples=10000
):
    # price1 = MC based on BSM
    rand_st = np.random.get_state() # Store random state first
    price1 = basket_price_mc(
        strike, spot, vol, weights, texp, cor_m,
        intr, divr, cp, True, n_samples)
    
    ''' 
    
    compute price2: mc price based on normal model
    make sure you use the same seed

    # Restore the state in order to generate the same state
    np.random.set_state(rand_st)  
    price2 = basket_price_mc(
        strike, spot, spot*vol, weights, texp, cor_m,
        intr, divr, cp, False, n_samples)
    '''
    price2 = basket_price_mc(
    strike, spot, vol, weights, texp, cor_m,
    intr=0.0, divr=0.0, cp=1, bsm=True, n_samples = 100000)

    ''' 
    compute price3: analytic price based on normal model
    
    price3 = basket_price_norm_analytic(
        strike, spot, vol, weights, texp, cor_m, intr, divr, cp)
    '''
    price3 = basket_price_norm_analytic(strike, spot, vol, weights, texp, cor_m, intr, divr, cp)
    
    # return two prices: without and with CV
    return np.array([price1, price1 - (price2 - price3)])


def basket_price_mc(
    strike, spot, vol, weights, texp, cor_m,
    intr=0.0, divr=0.0, cp=1, bsm=True, n_samples = 100000
):
    basket_check_args(spot, vol, cor_m, weights)
    
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    cov_m = vol * cor_m * vol[:,None]
    chol_m = np.linalg.cholesky(cov_m)  # L matrix in slides

    n_assets = spot.size
    znorm_m = np.random.normal(size=(n_assets, n_samples))
#----------------------------------------------------------------------------
    if bsm == True:
        '''
        PUT the simulation of the geometric brownian motion below
       # implement a RN generation for the asset   
       #how to simulate N asset prices at the same time

        '''
        prices = np.ones((n_assets,n_samples))
        for k in range(0,n_assets):
            prices[k] = forward[:,None][k] * (np.exp(-0.5 * texp * cor_m[k,k] \
                     + np.sqrt(texp)* (chol_m[k] @ znorm_m)))
#----------------------------------------------------------------------------
            
    else:
        # bsm = False: normal model
        prices = forward[:,None] + np.sqrt(texp) * chol_m @ znorm_m
    
    price_weighted = weights @ prices
#     print(znorm_m)
#     print(forward)
#     print(forward[:,None])
    print(prices)

    price = np.mean( np.fmax(cp*(price_weighted - strike), 0) )
    return disc_fac * price


def basket_price_norm_analytic(
    strike, spot, vol, weights, 
    texp, cor_m, intr=0.0, divr=0.0, cp=1
):
    '''
    The analytic (exact) option price under the normal model
    
    1. compute the forward of the basket
    2. compute the normal volatility of basket
    3. plug in the forward and volatility to the normal price formula
    

    
    PUT YOUR CODE BELOW
    '''
#----------------------------------------------------------------------------
    basket_check_args(spot, vol, cor_m, weights)
    n_assets = spot.size
    var_sum = 0
    for i in range(0,n_assets):
        for j in range(0,n_assets):
            if i == j :
                var_sum = var_sum + cor_m[i,i] * weights[i]**2
            else :
                var_sum = var_sum + cor_m[i,j] * weights[i] * weights[j] 
    bask_vol = var_sum * texp
    sigma = np.sqrt(bask_vol)
    norm = pf.Norm(sigma, intr=intr, divr=divr)
    price = norm.price(strike, spot, texp, cp=cp) 
#----------------------------------------------------------------------------    

  
    
    return price
