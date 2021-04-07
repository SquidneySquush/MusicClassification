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

def init_centroids (data, k): # data is the array of arrays and k is the number of clusters
    # get the number of columns (for fdata =2)
    n = np.shape(fdata)[1]
    # create empty centroid matrix 
    centroids = np.mat(np.zeros((k,n))) #np.zeros creates and ndarray -> np.mat turns it into a matix
    
    for i in range(n):
        min_i = min(fdata[:,i])
        max_i = max(fdata[:,i])
        # assign centroids random value that lies btwn min and max of that column
        centroids[:,i] = np.random.uniform(min_i, max_i, size = (k,1))

    return centroids

def Euclidean_distance (pt_one, pt_two):
    sq_dist = 0
    for i in range(len(pt_one)):
        sq_dist += (pt_one[i]-pt_two[i])**2
    
    eDist= math.sqrt(sq_dist)

    return(eDist)

#initial_centroids(fdata, 3)

def cluster(data, k):
    # number of rows in data array
    m= np.shape(data)[0] 
    print(m)
    # instance cluster assignments
    c_assignments = np.mat(np.zeros((m, 2))) 
    print(c_assignments)
    # initialize centroids
    centroids = init_centroids(fdata, k)

    #copy of original centroids
    centroids_og = centroids.copy()

    changed = True
    counter = 0

    # Loop until no cluster assignment changes
    while changed:

        changed = False
        
        # for every instance (row in dataset)
        for i in range(m):

            # Keep track of the minimum distance and minimum index for data
            min_dist = np.inf
            min_index = -1

            for l in range(k):
                # for every row (l) in the centroids matrix, calculate the 
                # distance between row (i) in data 
                dist_li = Euclidean_distance(centroids[l,:], data[i,:]) 
                # if the calc distance is less then our tracked min_dist 
                # set the min_dist to the calc dist and the min indext to 
                # the current row (l)
                if dist_li < min_dist:
                    min_dist = dist_li
                    min_index = l

            # if our current row of c_assignments does not equal the min_index
            # after the for loop then we change the c_assignments
            if c_assignments[i,0] != min_index:
                changed = True

            # assign point to cluster and list distance
            c_assignments[i, :] = min_index, min_dist

        # find new centroid
        for cent in range(k):
            point = 


cluster(fdata, 3)