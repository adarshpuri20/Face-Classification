# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 01:54:04 2020

@author: Adarsh Puri
"""
import matplotlib.pyplot as plt
import numpy as np

class roc(object):
    
    def __init__(self,shape):
        '''foooooooooo'''
    
    def ConfusionMatrix(self,pr): 
        '''Takes in a dataframe and gives list'''
        '''pr is Prediction Result'''
        add = pr.IsFace + pr.Predictions
        subtract = pr.IsFace - pr.Predictions
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0
        for i in range(len(add)):
            if add[i] == 0:
                true_neg = true_neg+1
            if add[i] == 2:
                true_pos = true_pos+1
            if subtract[i] == -1:
                false_pos = false_pos+1
            if subtract[i] == 1:
                false_neg = false_neg+1
        return true_pos,true_neg,false_pos,false_neg
    
    def calculate_TPR(self,pr): 
        '''pr is Prediction Result variable'''
        true_pos,true_neg,false_pos,false_neg = self.ConfusionMatrix(pr)
        return (true_pos/(true_pos+false_neg))
    
    def calculate_FPR(self,pr):
        true_pos,true_neg,false_pos,false_neg = self.ConfusionMatrix(pr)
        return (false_pos/(true_neg+false_pos))
                
    def probability_to_Predict(self,prediction_result,t):
        for k in range(len(prediction_result.Probability)):
            if prediction_result.Probability[k] <= t:
                prediction_result.at[k,'Predictions'] = 1
            else:
                prediction_result.at[k,'Predictions'] = 0
        return prediction_result
    
    def plot_ROC(self,prediction_result):
        TPR = []
        FPR = []
        traverse = np.linspace(0,1,100)
        for t1 in traverse:
            prediction_result = self.probability_to_Predict(prediction_result,t1)
            TPR.append(self.calculate_TPR(prediction_result))
            FPR.append(self.calculate_FPR(prediction_result))
        plt.plot(FPR,TPR,'b-')
        plt.plot([0, 1], [0, 1], 'k--')  # random guess predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()
        
        
    def evaluation_criteria(self,prediction_result):
        true_pos,true_neg,false_pos,false_neg = self.ConfusionMatrix(prediction_result)
        mis_class_rate = (false_pos + false_neg)/(true_pos+true_neg+false_pos+false_neg)
        F_N_R = (false_neg/(true_pos+false_neg))
        F_P_R = (false_pos/(false_pos+false_neg))
        print('Misclassification Rate = {}'.format(mis_class_rate))
        print('False Negative Rate = {}'.format(F_N_R))
        print('False Positive Rate = {}'.format(F_P_R))