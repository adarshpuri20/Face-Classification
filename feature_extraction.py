# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 01:29:16 2020

@author: Adarsh Puri
"""
import cv2
import numpy as np

#histogram_equalisation function(img,L):
#    img = img.astype(np.float64)
#    epsilon = 1e-6
#    if len(img.shape) == 2:
#        return (L*(img - min(img.flatten()))/(max(img.flatten())-min(img.flatten())+epsilon))
#    else:
#        img[:,:,0] = L*(img[:,:,0] - min(img[:,:,0].flatten()))/(max(img[:,:,0].flatten())-min(img[:,:,0].flatten())+epsilon)
#        img[:,:,1] = L*(img[:,:,1] - min(img[:,:,1].flatten()))/(max(img[:,:,1].flatten())-min(img[:,:,1].flatten())+epsilon)
#        img[:,:,2] = L*(img[:,:,2] - min(img[:,:,2].flatten()))/(max(img[:,:,2].flatten())-min(img[:,:,2].flatten())+epsilon)
#        return img

def feature_extraction(img = None, Shape = (20,20)):
    '''input Image and  output 400x1 vector'''
    if type(img) != type(None):
        if (img.shape[0],img.shape[1]) != Shape: #Resizing the image to feature vector
            img = cv2.resize(img,Shape,interpolation = cv2.INTER_AREA)
        #histogram equalization and  normalizing the image space to [0,1]
        #img = hist_equal(img,255)
        
        if len(img.shape) == 2:
            return img.flatten()
        else:
            output = np.zeros((400,1,3))
            output[:,:,0] = np.reshape(img[:,:,0].flatten(),(400,1))
            output[:,:,1] = np.reshape(img[:,:,1].flatten(),(400,1))
            output[:,:,2] = np.reshape(img[:,:,2].flatten(),(400,1))
            return np.reshape(output.flatten(),(1200,1))
    return None
        
    

def main():
    cv2.waitKey(0)
    print('the code is working')

if __name__ == '__main__':
    main()
    