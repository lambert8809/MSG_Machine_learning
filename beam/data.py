# Create database using LHS samlping.
# Usage: call load_data(parameter), parameter is the size of 
# samples in one dimension (same size in two dimensions)

from pyDOE import *
import numpy as np
import matplotlib.pyplot as plt

def load_data(sample_size):
    sampling = lhs(2,samples=sample_size,criterion='center')
    x = 10*sampling[:,0]
    q = 10*sampling[:,1]+5
    u = np.zeros(1000)
    k = np.zeros(1000) 
    Ei = 1e6
    L = 10
    for i in range(len(u)):
        u[i] = -1000*q[i]*L**4/24/Ei*(x[i]/L-x[i]**2/L**2)**2      # multiply 1000 to convert m to mm
        k[i] = -q[i]*(L**2-6*L*x[i]+6*x[i]**2)/12/Ei
    return x,q,u,k

if __name__ == "__main__":
    x,q,u,k = load_data(1000)
    # Plot data for a quick check
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(x, u)
    ax2 = f2.add_subplot(111)
    ax2.scatter(x, k)
    plt.show()
