# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 05:54:04 2020

@author: Adarsh Puri
"""
#import sys
#import os
import pandas as pd
import argparse
import cv2
import plot_ROC as roc
import mean_variance_plot as mvp
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
import feature_extraction as feature
import gaussian_model as gaussian_bin
#import plot_roc_curve_4
#import scipy
#import plot_mu_sig_3
import model_GMM as mog
import EM_init as em_init
import EM as em
import t_fit as tfit
from sklearn.decomposition import PCA
import pdf_tm as pdf_tm
import factoranalysis as f_a
import activationfuncsig as act
import mix_tfit as mit
import pdf_mixt

data_path = ('..//data_model//')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputcheck', action='store_true', help='data loading check.')
    parser.add_argument('--trainmog', action='store_true', help='train mixture of gaussian')
    parser.add_argument('--trainfactor', action='store_true',help='Factor Analysis model')
    parser.add_argument('--trainmodel', action='store_true',help='Training the model')
    parser.add_argument('--testmodel', action='store_true',help='Testing the model')
    parser.add_argument('--trainstud', action='store_true',help='train stud t distribution')
    parser.add_argument('--trainmixt', action='store_true',help='Training mixture of stud T')
    return parser.parse_args()

def load_data():
    dataframeface = pd.read_csv(data_path + 'face.csv')
    dataframetrainface = images_to_feature(dataframeface[0:1000].copy())
    dataframetestface = images_to_feature(dataframeface[1120:1220].copy())
    #dfTestFace
    dataframenonface = pd.read_csv(data_path + 'nonface.csv')
    dataframetrainnonface = images_to_feature(dataframenonface[0:1000].copy())
    dataframetestnonface = images_to_feature(dataframenonface[1120:1220].copy())
    dataframetrainface.to_csv('trainF_feature.csv',index = False)
    dataframetrainnonface.to_csv('trainNF_features.csv', index = False)
    
  
    print('Number of training data for face: {}'.format(len(dataframetrainface)))
    print('Number of testing data for face: {}'.format(len(dataframetestface))) 
    print('Number of training data for Nonface: {}'.format(len(dataframetrainnonface)))
    print('Number of testing data for Nonface: {}'.format(len(dataframetestnonface)))

    return dataframetrainface, dataframetestface, dataframetrainnonface, dataframetestnonface

def images_to_feature(df):
    'takes a dataframe and converts it to images'
    frame = []
    for i in df.Image:
       # print(os.path
        image = cv2.imread(data_path + i)
        frame.append(feature.feature_extraction(img = image))
        
        
    df['Feature'] = frame     
    
    return df

def gaussian():
    trainface,testface,trainnonface,testnonface = load_data()
    model = gaussian_bin.gaussianmodel_classifier((1200,1))
    model.estimate_parameter(trainface,trainnonface)
    df = pd.concat([testface,testnonface],ignore_index = True)
    store_pred,store_prob = model.predictions(df)
    df['Predictions'] = store_pred
    df['Probability'] = store_prob
    df.to_csv('prediction_gaussian.csv')
    r_o_c = roc.roc(1200)
    r_o_c.evaluation_criteria(df)
    r_o_c.plot_ROC(df)
    
    mvp.mean_variance_plot(model.mean_class_f,model.var_class_f,'Mean from Face Data','Covariance from Face data','RGB',20,'BOTH')
    mvp.mean_variance_plot(model.mean_class_nf,model.var_class_nf,'Mean from Non Face Data','Covariance from Non Face data','RGB',20,'BOTH')


############################################################################################################################################
def mixturegaussian():
    trainface,testface,trainnonface,testnonface = load_data()
    trainf_pdf,testf_pdf,trainnf_pdf,testnf_pdf= data_inpdf(trainface),data_inpdf(testface),data_inpdf(trainnonface),data_inpdf(testnonface)
   # priors , mu, sigma = EM_init.EM_init(trainf.Feature.to_numpy(),10)
    print(trainf_pdf.shape)
    print(trainnf_pdf.shape)
   
    prioris_info , mu_mean, covar = em_init.EM_init(trainf_pdf,3)
    checkdf = pd.DataFrame(prioris_info)
    checkdf.to_csv('The_initial_priors.csv')
    mudf = pd.DataFrame(mu_mean)
    mudf.to_csv('InitialMu.csv')
#    prioris_info,mu_mean,covar = em.EM(trainf_pdf,prioris_info,mu_mean,covar,iterate = 2)    
#    print('Prioris {}'.format(np.size(prioris_info)))
#    print('Mean {}'.format(mu_mean.shape))
#    print('Covariance {}'.format(covar.shape
   
    #MOG
    mog_f = mog.gmm_gmm(3)
    mog_f.fit(trainf_pdf,3)
    print('Training Done for Face')
    print(mog_f.muu.shape)
    print(mog_f.sig.shape)
    atmudf = pd.DataFrame(mog_f.muu)
    atmudf.to_csv('aftermean.csv')
    afprior = pd.DataFrame(mog_f.prior)
    afprior.to_csv('finalprrior.csv')
    mog_nf = mog.gmm_gmm(3)
    mog_nf.fit(trainnf_pdf,3)
    df = pd.concat([testface,testnonface],ignore_index = True)
    tnp = np.concatenate((testf_pdf,testnf_pdf),axis = 1)
    save_pred,save_prob = mog.mixofg_bin_classifier(tnp,mog_f.muu,mog_nf.muu,mog_f.sig,mog_nf.sig,mog_f.prior,mog_nf.prior)
    df['Predictions'] = save_pred
    df['Probability'] = save_prob
    df.to_csv('prediction_MOG.csv')
    r_o_c = roc.roc(1200)
    r_o_c.evaluation_criteria(df)
    r_o_c.plot_ROC(df)
    for i in range(mog_f.nbmixtures_states):
        mvp.mean_variance_plot(mog_f.muu[:,i],mog_f.sig[:,:,i],'mogfacemean','mogfacevar','RGB',20,'BOTH')
        mvp.mean_variance_plot(mog_nf.muu[:,i],mog_nf.sig[:,:,i],'mognonfacemean','mogfacevar','RGB',20,'BOTH')

def data_inpdf(dat):
    pdfdata=np.zeros((dat.Feature.to_numpy()[0].shape[0],(dat.Feature.to_numpy().shape[0])))
    for zeta in range(pdfdata.shape[1]):
        pdfdata[:,zeta]= np.reshape(dat.Feature.to_numpy()[zeta],(dat.Feature.to_numpy()[zeta].shape[0]))
    return pdfdata


########################################################################################################3
#from sklearn.preprocessing import MinMaxScaler
def studt():
    
    trainface,testface,trainnonface,testnonface=load_data()
    pca_face1=PCA(30)
    pca_face2=PCA(30)
    pca_face3=PCA(30)
    temp1=trainface.Feature.to_numpy()
    temp2=trainnonface.Feature.to_numpy()
    array_trainf = np.zeros((len(temp1),temp1[0].shape[0]))
    array_trainnf = np.zeros((len(temp2),temp2[0].shape[0]))
    for i in range(len(temp1)):
        array_trainf[i,:] = temp1[i].flatten()
        array_trainnf[i,:] = temp2[i].flatten()
        
    f_train_pca=pca_face1.fit_transform(array_trainf)
    nf_train_pca=pca_face2.fit_transform(array_trainnf)
    
    temp3=testface.Feature.to_numpy()
    temp4=testnonface.Feature.to_numpy()
    array_testf = np.zeros((len(temp3),temp3[0].shape[0]))
    array_testnf = np.zeros((len(temp4),temp4[0].shape[0]))
    for i in range(len(temp3)):
        array_testf[i,:] = temp3[i].flatten()
        array_testnf[i,:] = temp4[i].flatten()
        
    f_test_pca=pca_face3.fit_transform(array_testf)
    nf_test_pca=pca_face3.fit_transform(array_testnf)
    
    (mu_f,sig_f,nu_f)=tfit.fit_t(f_train_pca,0.01)
    (mu_nf,sig_nf,nu_nf)=tfit.fit_t(nf_train_pca,0.01)
                               
    px_face_pf = pdf_tm.pdf_tm(f_test_pca,mu_f,sig_f,nu_f)
    px_nonface_pf = pdf_tm.pdf_tm(nf_test_pca,mu_f,sig_f,nu_f)
    px_face_pnf = pdf_tm.pdf_tm(f_test_pca,mu_nf,sig_nf,nu_nf)
    px_nonface_pnf = pdf_tm.pdf_tm(nf_test_pca,mu_nf,sig_nf,nu_nf)
    
    Prob_face = np.concatenate((px_face_pf,px_nonface_pf))
    Prob_nonface = np.concatenate((px_face_pnf,px_nonface_pnf))
    
    
    
    df = pd.concat([testface,testnonface],ignore_index = True)
    #print(Prob_face.shape)
    pred = []
    for i in range(Prob_face.shape[0]):
        if Prob_face[i]-Prob_nonface[i] > 0:
            pred.append(1)
        else:
            pred.append(0)
    Prob_faced = Prob_face/(Prob_face+Prob_nonface)
    Prob_nonfaced = Prob_nonface/(Prob_face+Prob_nonface)
    df['CheckProbFace'] = Prob_faced
    df['CheckProbNonface'] = Prob_nonfaced        
    df['Predictions'] = pred
    df['Probability'] = act.sigmoid(5*(Prob_nonfaced-Prob_faced))
    
    df.to_csv('studentT.csv')
    r_o_c = roc.roc(1200)
    r_o_c.evaluation_criteria(df)
    r_o_c.plot_ROC(df)
    
    mu_f = pca_face1.inverse_transform(mu_f)
    mu_f=mu_f.flatten()
    mu_nf=pca_face2.inverse_transform(mu_nf)
    mu_nf=mu_nf.flatten()
    sig_f = pca_face1.inverse_transform(sig_f)
    sig_nf = pca_face2.inverse_transform(sig_nf)

    
    mvp.mean_variance_plot(mu_f,sig_f,'studfacemean','studfacevar','RGB',20,'BOTH')
    mvp.mean_variance_plot(mu_nf,sig_nf,'studnonfacemean','studnonfacevar','RGB',20,'BOTH')
    

##################################################################################################
   
    '''mixture of T'''
    '''Uncomment mix_tfit import file from top to run this part'''
    
def data_pca(temp1):
    data=temp1.Feature.to_numpy()
    array_data = np.zeros((len(data),data[0].shape[0]))
    for i in range(len(data)):
        array_data[i,:] = data[i].flatten()
       
    pca_face=PCA(30)    
    data_pca_format=pca_face.fit_transform(array_data)
    return data_pca_format,array_data

def mixoft():
    
    trainface,testface,trainnonface,testnonface=load_data()
    pca_face=PCA(30)
    f_train_pca,array_trainf = data_pca(trainface)
    f_test_pca,array_testf = data_pca(testface)
    nf_train_pca,array_trainnf = data_pca(trainnonface)
    nf_test_pca,array_testnf = data_pca(testnonface)

    x = f_train_pca
    K = 3
    [means_f,sig_f,nu_f,lam_f] = mit.fit_mix_t(x, 0.01,K)
    x = nf_train_pca
    [means_nf,sig_nf,nu_nf,lam_nf] = mit.fit_mix_t(x, 0.01,K)
    print(means_f)
    for k in range(K):
        m = pca_face.inverse_transform(means_f[k]) 
        s = pca_face.inverse_transform(np.diag(sig_f[k])) 
        X_proj_img = np.reshape(m,(20,20))
        X_proj_img = X_proj_img / np.max(m) 
        X_proj_img1 = np.reshape(s,(20,20))
        X_proj_img1 = X_proj_img1 / np.max(s)
        plt.imshow(X_proj_img,cmap='gray')   
        plt.show()
        plt.imshow(X_proj_img1,cmap='gray')    
        plt.show()
#        plot_mu_sig_3.plot_mu_sig(m,(s),'Mean of Face Images_'+str(k),'Variance of Face Images_'+str(k),'GRAY',60,'BOTH')
    #################################
    '''ROC plot mix of T'''
    '''shuffling test data'''
    n1,m1 = array_testf.shape
    n2,m2 = array_testnf.shape
    t0 = np.ones((n1,1))
    u0 = np.zeros((n2,1))
    tnew = np.hstack((array_testf,t0))
    unew = np.hstack((array_testf,u0))
    X = np.concatenate((tnew, unew), axis=0)
    np.random.shuffle(X[0:n1+n2])
    labels = X[:,-1]
    X_test_roc = X[:,:-1]
    X_roc = pca_face.fit_transform(X_test_roc)
    I,D = X_roc.shape
    temp = np.zeros([I,K])
    l2 = np.zeros([I,K])  
    
    for k in range(K):
        l2[:,k] = pdf_mixt.pdf_tm(X_roc, means_f[k,:], sig_f[k],nu_f[k])
        a = lam_f[k]
        temp[:,k] = a * l2[:,k];   
    Pr_face = np.sum(temp,axis=1);
    
    temp = np.zeros([I,K]);
    l2 = np.zeros([I,K]);  
    for k in range(K):
        l2[:,k] = pdf_mixt.pdf_tm(X_roc, means_nf[k,:], sig_nf[k],nu_nf[k])
        a = lam_nf[k]
        temp[:,k] = a * l2[:,k];           
    Pr_noface = np.sum(temp,axis=1);
    
    P_Roc = Pr_face / (Pr_face + Pr_noface)
#   plot_roc_curve_4.plot_roc_curve(labels,P_Roc,1,'Receiver Operating Characteristic for Mixture of T model')
    I,D = f_test_pca.shape
    temp = np.zeros([I,K]);
    l2 = np.zeros([I,K]);  
    for k in range(K):
        l2[:,k] = pdf_mixt.pdf_tm(f_test_pca, means_f[k,:], sig_f[k],nu_f[k])
        a = lam_f[k]
        temp[:,k] = a * l2[:,k];   
    px_face_pf = np.sum(temp,axis=1);
    
    temp = np.zeros([I,K]);
    l2 = np.zeros([I,K]);  
    for k in range(K):
        l2[:,k] = pdf_mixt.pdf_tm(f_test_pca, means_nf[k,:], sig_nf[k],nu_nf[k])
        a = lam_nf[k]
        temp[:,k] = a * l2[:,k];           
    px_face_pnf = np.sum(temp,axis=1)
    total = px_face_pf + px_face_pnf
    Prob_face = px_face_pf / total
    
    I,D = nf_test_pca.shape
    temp = np.zeros([I,K]);
    l2 = np.zeros([I,K]);  
    for k in range(K):
        l2[:,k] = pdf_mixt.pdf_tm(f_test_pca, means_f[k,:], sig_f[k],nu_f[k])
        a = lam_f[k]
        temp[:,k] = a * l2[:,k];   
    px_nonface_pf = np.sum(temp,axis=1);
     
    temp = np.zeros([I,K]);
    l2 = np.zeros([I,K]);  
    for k in range(K):
        l2[:,k] = pdf_mixt.pdf_tm(f_test_pca, means_nf[k,:], sig_nf[k],nu_nf[k])
        a = lam_nf[k]
        temp[:,k] = a * l2[:,k];           
    px_face_pnf = np.sum(temp,axis=1)
    total = px_face_pf + px_face_pnf
    px_nonface_pnf = px_face_pf / total
    total = px_nonface_pf + px_nonface_pnf
    Prob_nonface = px_nonface_pnf/ total
    
    cor_face_t = np.sum(Prob_face[:] >= 0.5)
    noncor_face_t = 100 - cor_face_t 
    cor_nonface_t = np.sum(Prob_nonface[:] >= 0.5)
    noncor_nonface_t = 100 - cor_nonface_t
    
    FPR_t = noncor_nonface_t / (noncor_nonface_t + cor_nonface_t)
    FNR_t =  noncor_face_t / (cor_face_t + noncor_face_t)
    MR = (noncor_nonface_t + noncor_face_t) / 100
    
    print('False Positive Rate:' + str(FPR_t))
    print('False Negative Rate:' + str(FNR_t))
    print('Miss Classification Rate:' + str(MR))
##########################################################################################################################3
                    #Factor Analysis
def factoranalysis():
    trainface,testface,trainnonface,testnonface = load_data()
    trainf_pdf,testf_pdf,trainnf_pdf,testnf_pdf = data_inpdf(trainface),data_inpdf(testface),data_inpdf(trainnonface),data_inpdf(testnonface)
    
    factorface = f_a.fact_analysis(3)
    factorface.fafit(trainf_pdf,3)    
    factornonface = f_a.fact_analysis(3)
    factornonface.fafit(trainnf_pdf,3)
    df = pd.concat([testface,testnonface],ignore_index = True)
    tnp = np.concatenate((testf_pdf,testnf_pdf),axis = 1)
    save_pred,save_prob = f_a.factorclassifier(tnp,factorface.mu,factornonface.mu,factorface.sigma,factornonface.sigma,factorface.theta,factornonface.theta)
    df['Predictions'] = save_pred
    df['Probability'] = save_prob
    df.to_csv('prediction_FA.csv')

                #Plotting ROC Curve
    r_o_c = roc.roc(1200)
    r_o_c.evaluation_criteria(df)
    r_o_c.plot_ROC(df)

                # mu and covariance
    temp1=factorface.theta
    temp2=factornonface.theta
    temp11=factorface.sigma
    temp22=factornonface.sigma
    covarface = np.matmul(temp1,np.transpose(temp1)) + np.diag(temp11)
    covarnonface = np.matmul(temp2,np.transpose(temp2)) + np.diag(temp22)

    mvp.mean_variance_plot(factorface.mu,covarface,'factorfacemean','factorfacevar','RGB',20,'BOTH')
    mvp.mean_variance_plot(factornonface.mu,covarnonface,'factornonfacemean','factornonfacevar','RGB',20,'BOTH')

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.inputcheck:
        load_data()
    if FLAGS.testmodel:
        'test()'
    if FLAGS.trainfactor:
        factoranalysis()
    if FLAGS.trainmodel:
        gaussian()
    if FLAGS.trainmog:
        mixturegaussian()
    if FLAGS.trainstud:
        studt()
    if FLAGS.trainmixt:
        mixoft()
