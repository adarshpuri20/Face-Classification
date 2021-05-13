import pdfforgaussian as pdfforgaussian
import numpy as np
import sys
#import pandas as pd
import scipy.special as sp
def EM(data,prior0,mu0,sig0,iterate = 10):
    '''Data is a numpy arra
    Input Parameters:
    Data is a np array of (1200,1000)
    prior0 is a np array of (1,10)
    mean is np array of (1200,10)
    var is a np array of (1200,1200,10)'''
    realmax = sys.float_info[0]
    realmin = sys.float_info[3]
    loglik_threshold = 1e-10
    prior = prior0
    mu = mu0
    sig = sig0
    loglik_old = realmax
   
    Pij = np.ndarray((data.shape[1],prior.shape[1]))
    lPij = np.ndarray((data.shape[1],prior.shape[1]))
   
    for zeta in range(iterate):
        #E-Step:   
        for j in range(prior.shape[1]):
            lPij[:,j] = np.nan_to_num((np.log(prior[0,j]+realmin) + pdfforgaussian.logGaussPdf(data,mu[:,j].reshape(mu.shape[0],1),sig[:,:,j])).reshape((data.shape[1])))
        Pij = sp.softmax(lPij,axis=1)


#        total_prob =np.sum(Pij,axis=1)+realmin
#        total_prob = total_prob.reshape((total_prob.shape[0],1))
#        total_prob = np.broadcast_to(total_prob,Pij.shape)
#        Pij = np.divide(Pij,total_prob)
#        df = pd.DataFrame(Pij)
#        df.to_csv('CheckPrior.csv')
        #M-Step:
        for j in range(prior.shape[1]):
            prior[0,j] = (np.sum(Pij[:,j]))/Pij.shape[0]
            mu[:,j] = (np.sum(np.multiply(np.broadcast_to(Pij[:,j],data.shape),data),axis = 1))/(np.sum(Pij[:,j])+realmin)
            a = data-mu[:,j].reshape(mu[:,j].shape[0],1)           
            print(a.shape)#1200,1000
            #sigsum = np.zeros((sig[:,:,0].shape))
#            for i in range(a.shape[1]):
#                sigsum = sigsum + Pij[i,j]*np.matmul(a[:,i].reshape(a.shape[0],1),np.transpose(a[:,i].reshape(a.shape[0],1)))
#            sig[:,:,j] = sigsum/np.sum(Pij[:,j])  
       
        loglik_new = 0
        for j in range(prior.shape[1]):
            loglik_new = loglik_new + prior[0,j]*pdfforgaussian.logGaussPdf(data,mu[:,j].reshape(mu.shape[0],1),sig[:,:,j])
        loglik = np.sum(loglik_new)
        
        prior = np.nan_to_num(prior)
        mu = np.nan_to_num(mu)
        sig = np.nan_to_num(sig)
        
        if np.absolute(loglik-loglik_old) <= loglik_threshold:
            return prior,mu,sig
        loglik_old = loglik
   
    return prior,mu,sig ####