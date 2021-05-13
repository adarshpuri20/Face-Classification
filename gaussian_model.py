# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:40:43 2020

@author: Adarsh  Puri
"""
import numpy as np

import activationfuncsig as act

class gaussianmodel_classifier(object):
    
    def __init__(self,shape):
        
        self.shape = shape
        self.mean_class_f = np.zeros(shape)
        self.var_class_f = np.matmul(self.mean_class_f,np.transpose(self.mean_class_f))
        self.mean_class_nf = np.zeros(shape)
        self.var_class_nf = np.matmul(self.mean_class_nf,np.transpose(self.mean_class_nf))
        
        
    def estimate_parameter(self,trainingdata_class_f,trainingdata_class_nf):
        '''inputs pandas dataframe and using that it will estimate the mean and variance'''
        
        self.mean_class_f = self.estimate_mean(trainingdata_class_f)
        self.var_class_f = self.estimate_variance(trainingdata_class_f,self.mean_class_f)
        self.mean_class_nf = self.estimate_mean(trainingdata_class_nf)
        self.var_class_nf = self.estimate_variance(trainingdata_class_nf,self.mean_class_nf)
        
        
        '''Training data is taken as a list of feature vectors'''

    def estimate_mean(self,x):
        '''Takes in the a pandas dataframe of feature vectors and returns the mean vector'''
        add = np.zeros((x.Feature[0].shape))
        for i in range(len(x.Feature)):
            add = add + x.Feature[i]
        return add/len(x.Feature)
    
    def estimate_variance(self,x,mu_mean):
        '''Takes in a pandas dataframe, estimated mean vector and returns a diagonal variance matrix of type array'''
        square_sum = np.zeros(x.Feature[0].shape)
        for j in range(len(x.Feature)):
            square_sum = (x.Feature[j]-mu_mean)**2
        diagmat = np.zeros(square_sum.shape)
        diagmat = np.matmul(diagmat,np.transpose(diagmat))
        for j in range(square_sum.shape[0]):
            diagmat[j][j] = square_sum[j]
        return diagmat/(len(x.Feature)-1)
    
    def predictions(self,data, threshold = 0.5):
        'Takes in a pandas dataframe and returns a list of classified output' 
        predic = []
        probab = []
        zeta1 = self.logrithmic_determinant_of_diagonalmat(self.var_class_f)
        zeta2= self.logrithmic_determinant_of_diagonalmat(self.var_class_nf)
        zeta=zeta1-zeta2
        for k in range(len(data.Feature)):
            predvalue, probvalue = self.classifier(data.Feature[k],zeta,threshold)
            predic.append(predvalue)
            probab.append(probvalue)
        return predic,probab
            
        '''to Calculate the logarithmic deteminant of a diagonal Matrix we take -----
        -------------we take input a 2-D symmetric array and returns a number Uses '''
    def logrithmic_determinant_of_diagonalmat(self,input_matrix):
        
        logdet = 0
        dim1=input_matrix.shape[0]
        for i in range(dim1):
            logdet = logdet + np.log(input_matrix[i][i]+1) 
        return logdet
            

    def classifier(self,featurevector, a1, t1):
               
        temp1 = np.matmul(np.transpose(featurevector-self.mean_class_f),np.linalg.inv(self.var_class_f)@(featurevector-self.mean_class_f))
        temp2 = np.matmul(np.transpose(featurevector-self.mean_class_nf),np.linalg.inv(self.var_class_f)@(featurevector-self.mean_class_nf))
        normalization_constant = 1e9 # a normalization constant to handle data overflow and underflow
        temp3=(act.sigmoid((a1+temp1-temp2)/normalization_constant))
        print(temp3)
        #print(temp2)
        #print(type(temp2))
        #print(a1)
        #print(type(a1))
        if temp3 <= t1:
            return 1,temp3
        else:
            return 0,temp3
    