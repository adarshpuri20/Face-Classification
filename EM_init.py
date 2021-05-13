import numpy as np

def kMeans(X, K, maxIters = 30):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return np.array(centroids) , C

def EM_init(data, nbStates):
    nbVar, nbData = np.shape(data)
    prioris = np.ndarray(shape = (1, nbStates))
    sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))
    Centers, Data_id = kMeans(np.transpose(data), nbStates)
    mu_mean = np.transpose(Centers)
    for i in range (0,nbStates):
        idtmp = np.nonzero(Data_id==i)
        idtmp = list(idtmp)
        idtmp = np.reshape(idtmp,(np.size(idtmp)))
        prioris[0,i] = np.size(idtmp)
        a = np.concatenate((data[:, idtmp],data[:, idtmp]), axis = 1)
        sigma[:,:,i] = np.cov(a)
        sigma[:,:,i] = sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))
    prioris = prioris / nbData
    return (prioris, mu_mean, sigma)
