import EM_init as kmean_init
import EM as em
import pdfforgaussian as gpdf
import numpy as np
#import pandas as pd
import activationfuncsig as act
import sys
import scipy.special as sp

realmin = sys.float_info[3]
class gmm_gmm(object):

    def __init__(self, nbmixtures_states):
        self.nbmixtures_states = nbmixtures_states
        self.muu=None
        self.sig=None
        self.prior=None

    def fit(self, data,iterations):
        prior, muu,sig = kmean_init.EM_init(data, self.nbmixtures_states)
        self.prior, self.muu, self.sig = em.EM(data, prior, muu, sig,iterate=iterations)

def mixofg_bin_classifier(y,mean1,mean2,sig1,sig2,pri1,pri2,th=0):
        temp1= np.ndarray((y.shape[1],pri1.shape[1]))
        temp2= np.ndarray((y.shape[1],pri1.shape[1]))
        
        for i in range(pri1.shape[1]):
            temp1[:,i] = np.nan_to_num(np.log(pri1[0,i]+realmin) + gpdf.logGaussPdf(y,mean1[:,i].reshape(mean1.shape[0],1),sig1[:,:,i])).reshape((temp1.shape[0]))
        for i in range(pri2.shape[1]):
            temp2[:,i] = np.nan_to_num(np.log(pri2[0,i] +realmin) +gpdf.logGaussPdf(y,mean2[:,i].reshape(mean2.shape[0],1),sig2[:,:,i])).reshape((temp2.shape[0]))
        
        pred=[]
        
        temp1 = np.nan_to_num(sp.logsumexp(temp1,axis=1))
        temp2 = np.nan_to_num(sp.logsumexp(temp2,axis=1))
        
        Prob1 = temp1/(temp1+temp2)
        Prob2 = temp2/(temp1+temp2)
        Prob1 = Prob1.reshape((temp1.shape[0]))
        Prob2 = Prob2.reshape((temp2.shape[0]))
        
        for i in range(Prob1.shape[0]):
            if (Prob2[i]-Prob1[i]) > 0:
                pred.append(1)
            else:
                pred.append(0)
        
        return pred, act.sigmoid(100*(Prob1-Prob2))
        
        
        
def predictions(x,mean1,mean2,sig1,sig2,pri1,pri2,th=0):
        pred = []
        prob=[]
        for i in range(len(x)):
            val_predict,val_prob = mixofg_bin_classifier(x[:,i],mean1,mean2,sig1,sig2,pri1,pri2,th)
            pred.append(val_predict)
            prob.append(val_prob)
        return pred.prob
    