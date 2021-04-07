import numpy as np
#import panda as pd
import math
import random

fdata = np.array([[.2,.3], [.3,.4], [.4,.3], [.3,.8], [.7,.7],[.8,.7],[.8,.9]])
#idk = np.asmatrix(fdata) #turns into matrix
# find min of array
gen =  min(fdata, key = min)
fdmin = min(x for x in gen)
#####minel = np.amin(fdata) # doesnt work right?
print(gen, fdmin)
print (np.shape(fdata)[1])

def initial_centroids (data, k): # data is the array of arrays and k is the number of clusters
    # get the number of columns (for fdata =2)
    n = np.shape(fdata)[1]
    # create empty centroid matrix 
    centroids = np.mat(np.zeros((k,n))) #np.zeros creates and ndarray -> np.mat turns it into a matix
    print(type(centroids), centroids)

    for i in range(n):
        min_i = min(fdata[:,i])
        max_i = max(fdata[:,i])
        # assign centroids random value that lies btwn min and max of that column
        centroids[:,i] = np.random.uniform(min_i, max_i, size = (k,1))

    return centroids

def Euclidean_distance ():
    sq_dist = 0
    for i in range(len(ft_one)):
        sq_dist += (ft_one[i]-ft_two[i])**2
    
    eDist= math.sqrt(sq_dist)

    return(eDist)

initial_centroids(fdata, 3)