# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 04:52:56 2020

@author: adars
"""

import numpy as np
import fit_mixt_cost
import scipy 
import pdf_mixt
from numpy import matlib as mb
#
#def fit_mix_t(x, precision,K):
#    resp=np.zeros((K,1000))
#    #E Step    
#    D = x.shape[1]
#    exph=np.zeros((K,1000,1))
#    for i in range(1000):
#        num=0
#        den=0
#        for k in range(K):    
#            den += np.nan_to_num\
#                (weights[k]*create_t_dist(x[:,i].reshape(-1,1), means[k], covars[k], v[k]))
#        for k in range(K):
#            num = np.nan_to_num\
#                (weights[k]*create_t_dist(x[:,i].reshape(-1,1), means[k], covars[k], v[k]))
#            resp[k][i]=num/den
#        for k in range(K):
#            exph[k,i] = (v[k]+D)/ \
#                (v[k] + np.dot(np.dot((x[:,i].reshape(-1,1)-means[k].reshape(-1,1)).T,\
#                  np.linalg.inv(covars[k])),(x[:,i].reshape(-1,1)-means[k].reshape(-1,1))))
#    #M Step 
#    sum_rik = np.sum(resp, axis=1)
#    #update weights 
#    for k in range(K):
#        weights[k] = sum_rik[k]/np.sum(sum_rik)
#        # print(weights)
#    # update means
#    for k in range(K):
#        num=0
#        for i in range(1000):
#            num+=resp[k][i]*exph[k,i]*x[:,i].reshape(-1,1)
#        means[k] = num/(sum_rik[k]*np.sum(exph[k]))
#    # # Update cov
#    for k in range(K):
#        cov_temp = np.zeros((100,100))
#        for i in range(1000):
#            cov_temp+= resp[k][i]exph[k,i]\np.matmul((x[:,i].reshape(-1,1)-means[k]\
#                   .reshape(-1,1)),(x[:,i].reshape(-1,1)-means[k].reshape(-1,1)).T)
#        cov_temp/=sum_rik[k]
#        covars[k] = np.diag(np.diag(cov_temp))   
#    #update v
#    for k in range(K):
#         v[k]=fminbound(costFunction, 6, 10, args=(D, resp[k], exph[k]))         
#    return weights, means, covars, v1, exph
def fit_mix_t(x, precision,K):
    print('Performing Mixture of fit_t...')
    np.random.seed(0)
    (I,D) = np.shape(x)
    sig = [[]] * K
    lam = mb.repmat(1/K, K, 1)
    
    K_integers = np.random.permutation(I);
    K_integers = K_integers[0:K]
    means = x[K_integers,:] 
          
    dataset_cov = np.cov(x,rowvar=False, bias=1, ddof=None)
    dataset_cov = np.diagonal(dataset_cov)  
    dataset_variance = np.diag(dataset_cov,0) + (10**-6)
    
    for i in range (K):
        sig[i] = dataset_variance;
        
    ##Initialize degrees of freedom to 100 (just a random large value).
    nu = mb.repmat(10, K, 1)       
    ##The main loop.
    iterations = 0    
    loop=3
    previous_L = 1000000 # just a random initialization
    for i in range(loop):
        tau = np.zeros([I,K])
        for k in range(K):
            temp_tau = pdf_mixt.pdf_tm(x,means[k,:],sig[k],nu[k])
            tau[:,k] = (lam[k] * temp_tau)
        tau_sum = np.sum(tau,axis = 1)
        tau = tau / np.reshape(tau_sum,(I,-1))
    
        
        delta = np.zeros([I,K])
        for i in range(I):
            for k in range(K):
                delta[i,k] = (scipy.spatial.distance.mahalanobis(np.reshape(x[i,:],(1,-1)),means[k],np.linalg.inv(sig[k]))**2)
    
        nu_plus_delta = np.zeros([I,K])
        E_hi = np.zeros([I,K])
        nu_plus_D = nu + D
        nu_plus_delta = np.transpose(nu) + delta
        for i in range(I):
            for k in range(K):
                E_hi[i,k] = np.divide(nu_plus_D[k],nu_plus_delta[i,k])
               
        ## Maximization step.
        ## Update lambda
        for k in range(K):
            lam = np.sum(tau,axis=0) / I
           
        E_hi_times_tau = E_hi * tau
        E_hi_times_tau_sum = np.sum(E_hi_times_tau,axis = 0)
        new_mu = np.zeros([K,D])
        for k in range(K):
            for i in range(I):
                new_mu[k,:] = new_mu[k,:] + (E_hi_times_tau[i,k]*x[i,:])
            means[k,:] = np.divide(new_mu[k,:],E_hi_times_tau_sum[k])
    
        #Update sig.
        tau_sum_col = np.sum(tau,axis=0)
        for k in range(K):
            new_sigma = np.zeros([D,D])
            for i in range(I):
                mat = np.reshape((x[i,:] - means[k,:]),(1,-1))
                mat = E_hi_times_tau[i,k] * (np.transpose(mat) * mat)
                new_sigma = new_sigma + mat
            sig[k] = np.divide(new_sigma,tau_sum_col[k]);
            sig[k] = np.diag(np.diag(sig[k]))
#                   
#        for k in range(K):      
#            nu[k] = fit_mixt_cost.fit_t_cost(nu[k],E_hi[:,k],D)
        for i in range(I):
            for k in range(K):
                delta[i,k] = (scipy.spatial.distance.mahalanobis(np.reshape(x[i,:],(1,-1)),means[k,:],np.linalg.inv(sig[k]))**2)
                
        iterations = iterations + 1;
        print(str(iterations))
        if iterations == 20:
            break
    return(means,sig,nu,lam)