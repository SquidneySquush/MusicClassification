from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import panda as pd
import math

from tempfile import TemporaryFile
import os
import pickle
import random 
import operator

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

k = 10 


def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0*correct/len(testSet)

directory = "genres/"
f= open("my.dat" ,'wb')
i=0

for folder in os.listdir(directory):
    i+=1
    print(folder)
    if i==11 :
        break 	
    for file in os.listdir(directory+folder):	
        (rate,sig) = wav.read(directory+folder+"/"+file)
        mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix , covariance , i)
        pickle.dump(feature , f)

f.close()

dataset = []
def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break	

    for x in range(len(dataset)):
        if random.random() <split :			
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])	

trainingSet = []
testSet = []
loadDataset("my.dat" , 0.66, trainingSet, testSet)  # Taking 66% of dataset


def initial_centroids(Features, k=clusters):
    n = np.shape(Features)[1]
    centroids = np.mat(np.zeros((k, n)))

    for i in range(n):
        min_i = min(Features[:,i])
        range_i = float(max(Features[:i])- min_i)
        centroids[:,i] = min_i + range_i * np.random.rand(k, 1)
    
    return centroids



def Euclidean_distance (ft_one, ft_two):
    sq_dist = 0
    for i in range(len(ft_one)):
        sq_dist += (ft_one[i]-ft_two[i])**2
    
    eDist= sqrt(sq_dist)

    return(eDist)

def cluster(Features, k , max_iter = 5985): 
    
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

            for j in range(k):

                dist_li = Euclidean_distance(centroids[l,:], Features[i,:])
                if dist_li < min_dist:
                    min_dist = dist_li
                    min_index = l

            if c_assignments[i,0] != min_index:
                changed = True

