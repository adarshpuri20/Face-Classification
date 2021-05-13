# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 04:54:23 2020

@author: adars
"""

from scipy.special import gammaln
import numpy as np


def pdf_tm(x, mu, sig, nu):
    D = len(mu);
    c = np.exp(gammaln((nu+D)/2) - gammaln(nu/2));
    c = c / (((nu*np.pi)**(D/2)) * np.sqrt(np.linalg.det(sig)));
    I = x.shape[0];
    delta = np.zeros([I,1]);
    x_minus_mu = np.subtract(x, mu)
    temp = np.dot(x_minus_mu,np.linalg.inv(sig))
    for i in range(I): 
        delta[i] = np.dot(temp[i,:],np.reshape(x_minus_mu[i,:],(-1,1)))
    px = 1 + (delta / nu)
    px = px**((-nu-D)/2)
    px = np.dot(px,c)
    return px