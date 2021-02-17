import matplotlib.pyplot as plt
import numpy as np 
from numpy.linalg import norm

def plotUnitBall(p, n_samples=5000):
    for i in range(n_samples):
        x = np.array([np.random.rand()*2-1,np.random.rand()*2-1]) #random point in [-1,1] x [-1,1]
        if norm(x,ord=p) <= 1:
            plt.scatter(x[0],x[1], color='blue')
    plt.axis('square')
    title = 'Unit %s ball' % str(p)
    plt.title(title, fontsize=16)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
    

def unitBallVolume(p, n_samples=10000):
    vol = 0
    for i in range(n_samples):
        x = np.array([np.random.rand()*2-1,np.random.rand()*2-1]) #random point in [-1,1] x [-1,1]
        if norm(x,ord=p) <= 1:
            vol += 1
    vol /= n_samples
    vol *= 4
    return vol