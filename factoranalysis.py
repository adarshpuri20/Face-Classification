# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 05:04:55 2020

@author: adars
"""

import faem as em
import pdfforgaussian as gpdf
import activationfuncsig as act
import pandas as pd
import json
import numpy as np
import scipy.special as sp


class fact_analysis(object):
   
    def __init__(self,numbfactor):
        '''Parameters:
        numbfactor: Number of factors'''
       
        self.numbfactor = numbfactor
        self.mu = None
        self.sigma = None
        self.theta = None
       
    def defaultdatainitialise(self,data):
        '''
        Self.sigma has a shape of 1200, and self.mu has a shape 1200,1
        '''
        self.mu = np.mean(data,axis = 1)
        x_minus_mean = np.subtract(data,self.mu.reshape(self.mu.shape[0],1))
        self.sigma = np.sum((x_minus_mean)**2,axis = 1)
        np.random.seed(0)
        self.theta = np.random.randn(data.shape[0],self.numbfactor)
       
       
    def fafit(self,data,loop):
        #Calculate the initial guess using Kmean method
        self.defaultdatainitialise(data)
        #Calculate the actual prior, mu and sigma
        self.mu,self.theta,self.sigma = em.expectaiton_maximization(data,self.mu,self.theta,self.sigma,loop=loop)
   
    def save(self,file1):
        save_data = {'Clusters':self.numbfactor,
                    'Mu':self.mu,
                    'Sigma':self.sigma,
                    'Prior':self.prior}
        f = open(file1,'w')
        json.dump(save_data,f)
        f.close()
       
       
def factorclassifier(x,m1,m2,s1,s2,theta1,theta2,threshold=0.5):
    '''
        threshold =0.5
        x: the input data=array(featurelength=1200,number of datapoints=1000)
        m1,m2: Mean of class face and non face, shape =(featurelength,)
        s1,s2 : Variance of the calss face and non face diagonal mat, shape=(1200,)
        theta1,theta2: factor matrix, shape=(1200,3)
    '''
    predict = []
    mu1 = m1.reshape(m1.shape[0],1)
    mu2 = m2.reshape(m2.shape[0],1)
    sigma1 = np.matmul(theta1,np.transpose(theta1)) + np.diag(s1)
    sigma2 = np.matmul(theta2,np.transpose(theta2)) + np.diag(s2)
   
    p1 = gpdf.logGaussPdf(x,mu1,sigma1)
    p2 = gpdf.logGaussPdf(x,mu2,sigma2)
    p1 = sp.softmax(p1)
    p2 = sp.softmax(p2)
    Prob1 = p1/(p1+p2)
    Prob2 = p2/(p1+p2)
    
    Prob1 = Prob1.reshape(p1.shape[0])
    Prob2 = Prob2.reshape(p2.shape[0])
    
#    
   
    for i in range(p1.shape[0]):
        if (Prob2[i]-Prob1[i]) <0:
            predict.append(1)
        else:
            predict.append(0)
    return predict, act.sigmoid(1000*(Prob2-Prob1))