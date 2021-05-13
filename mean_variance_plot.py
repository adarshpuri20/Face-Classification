# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 03:17:03 2020

@author: adars
"""

import cv2
import numpy as np

def mean_variance_plot (mu1,sigma1,title1,title2,img_type,image_size,ploting):
    # Plot Mean  
    mu = mu1 / np.max(mu1)
    if (img_type == 'GRAY'):
        mu_mat = np.reshape(mu,(image_size,image_size))
    else:
        mu_mat = np.reshape(mu,(image_size,image_size,3))
    r = 200.0 / mu_mat.shape[1]
    dim = (200, int(mu_mat.shape[0] * r))
    resized = cv2.resize(mu_mat, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("resized", resized)
    resized = resized*(255/np.max(resized))
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(title1+'.png',resized.astype('uint8'))
        cv2.destroyAllWindows()
     
    if (ploting == 'BOTH'):
        #% Plot covariane
        sigma = np.zeros(sigma1.shape[0])
        for k in range(sigma1.shape[0]):
            sigma[k] = sigma1[k][k]
        sigma = sigma/np.max(sigma)
        if (img_type == 'GRAY'):
            sigma_mat = np.reshape((sigma),(image_size,image_size))
        else:
            sigma_mat = np.reshape((sigma),(image_size,image_size,3))
        r = 200.0 / sigma_mat.shape[1]
        dim = (200, int(sigma_mat.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(sigma_mat, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("resized", resized)
        resized = resized*(255/np.max(resized))
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(title2 +'.png',resized)
            cv2.destroyAllWindows()
