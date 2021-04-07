from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
#import panda as pd
import math

from tempfile import TemporaryFile
import os
import pickle
import random 
import operator


direct = "genres/"
f= open("my.dat" ,'wb')
i=0

for folder in os.listdir(direct):
    if folder.endswith(".mf"):
        continue
    else:
        i+=1
        for file in os.listdir(direct+folder):	
            (rate,sig) = wav.read(direct+folder+"/"+file)
            mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            #Features = (mean_matrix , covariance , i)
           # print(feature)
          #  pickle.dump(Features , f)

f.close()

print(type(covariance))
print(type(covariance[1]),covariance[1])

def initial_centroids(Features, k):
    print(Features)
    n = np.shape(Features)[1]
    centroids = np.mat(np.zeros((k, n)))

    for i in range(n):
        min_i = min(Features[:,i])
        range_i = float(max(Features[:,i])- min_i)
        centroids[:,i] = min_i + range_i * np.random.rand(k, 1)
    
    return centroids



def Euclidean_distance (ft_one, ft_two):
    sq_dist = 0
    for i in range(len(ft_one)):
        sq_dist += (ft_one[i]-ft_two[i])**2
    
    eDist= math.sqrt(sq_dist)

    return(eDist)

def cluster(Features, k = 10 , max_iter = 5985): 
    
    m= np.shape(Features)[0] # # of rows in features

    c_assignments = np.mat(np.zeros((m, 2))) 

    centroids = initial_centroids(Features, k)

    centroids_og = centroids.copy() #copy of original centroids

    changed = True
    counter = 0

    while changed:

        changed = False

        for i in range(m):
            min_dist = np.inf
            min_index = -1

            for l in range(k):

                dist_li = Euclidean_distance(centroids[l,:], Features[i,:])
                if dist_li < min_dist:
                    min_dist = dist_li
                    min_index = l

            if c_assignments[i,0] != min_index:
                changed = True

            c_assignments[i, :] = min_index, min_dist**2 #assign point to cluster

            # find new centroid
        for centroid in range(k):
            point = Features[np.nonzero(c_assignments[:,0].A==centroid)[0]]
            centroids[centroid, :]= np.mean(point, axis=0)

        counter += 1

    return centroids, c_assignments, counter, centroids_og

        

#centroids, c_assignments, count, og_centroids = cluster(Features ,10)

