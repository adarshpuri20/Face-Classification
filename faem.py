# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:50:10 2020

@author: adars
"""

import numpy as np

def expectaiton_maximization(data,mu0,theta0,sig0,loop = 10):
    '''
    data = 1200*1000 array
    theta0 = 1200*number of unknown factors
    mu0 = 1200, sig0 = 1200, zeta =1000, beta= 1200
    kfactor = number of factor
    '''
    mu=mu0   
    theta = theta0
    covar = sig0
   
    (beta,zeta) = np.shape(data)
    kfactor = theta.shape[1]
    
    for light in range(loop):
       
        #E-Step:
        x_minus_mu = np.subtract(data,mu.reshape(mu.shape[0],1))
       
        inv_covar = np.diag(1 / covar)
        temp_product = np.dot(np.transpose(theta),inv_covar)
        temp = np.linalg.inv(np.dot(temp_product,theta) + np.identity(kfactor))
        E_hi = np.dot(np.dot(temp,temp_product),x_minus_mu)
        E_hi_hitr = [[]]*zeta
        for i in range (zeta):
            ee = E_hi[:,i]
            E_hi_hitr[i] = temp + np.dot(ee,np.transpose(ee))
       
        #M-step
        #Update Phi
        theta1 = np.zeros([beta,kfactor])
        for i in range(zeta):
            temp1 = x_minus_mu[:,i].reshape(x_minus_mu[:,i].shape[0],1)
            temp2 = np.transpose(np.reshape(E_hi[:,i],(-1,1)))
            theta1 = theta1 + np.dot(temp1,temp2)
       
        theta2 = np.zeros([kfactor,kfactor])
        for i in range(zeta):
            theta2 = theta2 + E_hi_hitr[i]
        theta2 = np.linalg.inv(theta2)
        theta = np.dot(theta1,theta2)
       
        #Updating the Covaraince
        covar_diagonal = np.zeros([beta,1])
        for i in range(zeta):
            xm = np.transpose(x_minus_mu[:,i])
            c1 = xm * xm
            c2 = np.dot(theta,E_hi[:,i]) * xm;
            covar_diagonal = covar_diagonal + c1 - c2
        covar = covar_diagonal / zeta
        covar = np.diag(covar)
       
       
    return mu,theta,covar
