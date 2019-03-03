# Create database using LHS samlping.
# Usage: call load_data(parameter), parameter is the size of 
# samples in one dimension (same size in two dimensions)

from pyDOE import *
import numpy as np
import matplotlib.pyplot as plt

def load_data(sample_size):
    sampling = lhs(2,samples=sample_size,criterion='center')
    idata = np.zeros((sample_size,2))           #[x,q]
    idata[:,0] = 10*sampling[:,0]
    idata[:,1] = 10*sampling[:,1]+5
    np.random.shuffle(idata)         # ramdon the samples for splitting
    trainD, validateD, testD = np.split(idata,[int(0.7*len(idata)),int(0.9*len(idata))]) 
    #x = 10*sampling[:,0]
    #q = 10*sampling[:,1]+5
    trainL = np.zeros((len(trainD),2))
    validateL = np.zeros((len(validateD),2))
    testL = np.zeros((len(testD),2))
    #u = np.zeros(1000)
    #k = np.zeros(1000) 
    Ei = 1e6
    L = 10
    for i in range(len(trainD)):
        # multiply 1000 to convert m to mm
        trainL[i,0] = -1000*trainD[i,1]*L**4/24/Ei*(trainD[i,0]/L-trainD[i,0]**2/L**2)**2    
        # multiply 1000 to convert the similar order of u    
        trainL[i,1] = -1000*trainD[i,1]*(L**2-6*L*trainD[i,0]+6*trainD[i,0]**2)/12/Ei
    for i in range(len(validateD)):
        # multiply 1000 to convert m to mm
        validateL[i,0] = -1000*validateD[i,1]*L**4/24/Ei*(validateD[i,0]/L-validateD[i,0]**2/L**2)**2  
        # multiply 1000 to convert the similar order of u   
        validateL[i,1] = -1000*validateD[i,1]*(L**2-6*L*validateD[i,0]+6*validateD[i,0]**2)/12/Ei
    for i in range(len(testD)):
        # multiply 1000 to convert m to mm
        testL[i,0] = -1000*testD[i,1]*L**4/24/Ei*(testD[i,0]/L-testD[i,0]**2/L**2)**2   
        # multiply 1000 to convert the similar order of u     
        testL[i,1] = -1000*testD[i,1]*(L**2-6*L*testD[i,0]+6*testD[i,0]**2)/12/Ei
    return trainD, validateD, testD, trainL, validateL, testL

if __name__ == "__main__":
    trainD, validateD, testD, trainL, validateL, testL = load_data(1000)
    # Plot data for a quick check
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(trainD[:,0], trainL[:,0])
    ax2 = f2.add_subplot(111)
    ax2.scatter(trainD[:,0], trainL[:,1])
    plt.show()
