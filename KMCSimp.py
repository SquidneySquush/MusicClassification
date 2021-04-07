import numpy as np
#import panda as pd
import math

fdata = np.array([[.2,.3], [.3,.4], [.4,.3], [.3,.8], [.7,.7],[.8,.7],[.8,.9]])


gen =  min(fdata, key = min)
fdmin = min(x for x in gen)

print(gen, fdmin)