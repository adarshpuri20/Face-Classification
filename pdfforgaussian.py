import numpy as np
import sys
import pandas as pd
import activationfuncsig as act
realmin = sys.float_info[3]
def pdfforgaussian(data, Mu, Sigma):
    
#    print(vector.shape)
#    print(Mu.shape)
#    print(Sigma.shape)
   
    return np.exp(logGaussPdf(data, Mu, Sigma))+realmin
   
def logGaussPdf(data,Mu,Sigma):
    
    nbVar = data.shape[0]
    try:
        nbData = data.shape[1]
    except:
        nbData = 1
    a = (-nbVar/2)*np.log(2*np.pi)
    (sign,logdet) = np.linalg.slogdet(Sigma)
    c = data-Mu
    x = np.zeros((nbData,1))
    var_inv = np.linalg.inv(Sigma) # + realmin*np.identity(Sigma.shape[0]))
    for i in range(nbData):
        unk = var_inv@((c[:,i]).reshape(c.shape[0],1))
        x[i,0] = np.dot(np.transpose(c[:,i].reshape((c.shape[0],1))),unk)
       
    var = pd.DataFrame(x)
    var.to_csv('var.csv')
   
   
    logpdf = a - (0.5)*logdet -(0.5*x)
    #print(logpdf.shape)
#    logpdfcheck = pd.DataFrame(logpdf)
#    logpdfcheck.to_csv('logpdf.csv')
   
    return logpdf

def sigmoid_classifier(data,Mu,Sigma):
    checksig = logGaussPdf(data,Mu,Sigma)
    df = pd.DataFrame(checksig)
    df.to_csv('checksig.csv')
    normalizer = 1e10
    valuereturn = act.sigmoid(checksig/normalizer)
    return valuereturn
