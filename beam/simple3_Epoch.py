import tensorflow as tf
import numpy as np
import pandas as pd
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
from data2 import *

def initialize_NN(layers):        
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = tf.Variable(tf.random_normal([layers[l], layers[l+1]], dtype=tf.float64), dtype=tf.float64)
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
        weights.append(W)
        biases.append(b)        
    return weights, biases

def net_u(X, weights, biases):
    num_layers = len(weights) + 1    
    H = X[:]
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
        #H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

def write_to_file(xt, pred_val,true_val):
    w=open("eval2.txt","w")
    for i in range(len(pred_val)):
        w.write(str(xt[i][0])+' '+str(pred_val[i][0])+'  '+str(true_val[i][0])+'\n')
    w.close()

def derivative_u(us,xs):
    us = us/1000
    u_x = tf.gradients(us, xs)[0]
    u_xx = tf.gradients(u_x, xs)[0]
    return u_xx

def derivative_M(Ms,xs):
    M_x = tf.gradients(Ms, xs)[0]
    M_xx = tf.gradients(M_x, xs)[0]
    return M_xx

def derivative_M1(Ms,xs):
    M_x = tf.gradients(Ms, xs)[0]
    return M_x

def derivative_u_x(us,xs):
    M_x = tf.gradients(us, xs, stop_gradients=[xs[:,1]])[0]
    return M_x


def L2Regluarize(weights):
    L2_temp = 0
    for i in range(len(weights)):
        #print("L2" + str(weights[i]))
        L2_reg = L2_temp + tf.nn.l2_loss(weights[i])
    return L2_reg

#layers = [1, 5, 5, 5,1]
# layers = [2, 100,100,100, 2] 
# layers2 =[2, 100,100,100, 2] 
layers = [2, 100,100,100, 2] 
layers2 =[2, 100,100,100, 2] 
# layers = [2, 40,40, 2] 
# layers2 =[2, 40,40, 2] 
#layers2 =[1,40, 40, 40, 1]
#layers = [1, 400, 400, 200, 200, 100, 1]

#large size data points
trainD, validateD, testD, trainL, validateL, testL = load_data(10000)
#small size data points
#trainD_t1, validateD_t1, testD_t1, trainL_t1, validateL_t1, testL_t1 = load_data(2000)
trainD_t1, validateD_t1, testD_t1, trainL_t1, validateL_t1, testL_t1 = load_data(3000)
# define variables and functions for large size data
weights, biases = initialize_NN(layers)
weights2, biases2 = initialize_NN(layers2)

#define the weights and biases for small size points
weights_t1, biases_t1 = initialize_NN(layers)
weights2_t1, biases2_t1 = initialize_NN(layers2)

input_0 = tf.placeholder(tf.float64,shape=[None,2])
output = tf.placeholder(tf.float64,shape=[None,2])

#Predict value of large size data
pred = net_u(input_0,weights,biases)
pred2 = net_u(input_0,weights2,biases2)

#Predict value of small size data
pred_t1 = net_u(input_0,weights_t1,biases_t1)
pred2_t1 = net_u(input_0,weights2_t1,biases2_t1)

#Constraint of first (large) set of data
u_pred_xx = derivative_u(tf.gather(pred2, 0, axis=1),input_0)

u = tf.gather(pred2, 0, axis=1)
k = pred2[:,1]

u_xx = u_pred_xx[:,0]

loss_constraint = tf.reduce_sum(tf.square(u_xx - k))

#Constraint of second(small) set of data
u_pred_xx_t1 = derivative_u(tf.gather(pred2_t1, 0, axis=1),input_0)

u_t1 = tf.gather(pred2_t1, 0, axis=1)
k_t1 = pred2_t1[:,1]

u_xx_t1 = u_pred_xx_t1[:,0]

loss_constraint_t1 = tf.reduce_sum(tf.square(u_xx_t1 - k_t1))
#u_pred2 = (x_u+1)*u_pred+(x_u-1)*u_pred
# Prototype "loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))"

#L2Regularizer of large size data
lossL2 = L2Regluarize(weights)*0.002
lossL2_2 = L2Regluarize(weights2)*0.002
#+ 0.0002*loss_constraint

#L2Regularizer of small size data
lossL2_t1 = L2Regluarize(weights_t1)*0.002
lossL2_2_t1 = L2Regluarize(weights2_t1)*0.002

#Loss function of large size data
l1 = tf.reduce_mean(tf.square(output - pred) + lossL2)
l2 = tf.reduce_mean(tf.square(output - pred2) + lossL2_2+ 0.0001*loss_constraint)

#Loss function of small size data
l1_t1 = tf.reduce_mean(tf.square(output - pred_t1) + lossL2_t1)
l2_t1 = tf.reduce_mean(tf.square(output - pred2_t1) + lossL2_2_t1+ 0.0001*loss_constraint_t1)

#l2 =  tf.reduce_mean(tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred))
#loss = tf.reduce_mean(tf.square(m-m_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.002)

#Train function of large large data
train_op1 = optimizer.minimize(l1)
train_op2 = optimizer.minimize(l2)
#train_op = optimizer.minimize(loss)

#Train function of small large data
train_op1_t1 = optimizer.minimize(l1_t1)
train_op2_t1 = optimizer.minimize(l2_t1)

init = tf.global_variables_initializer()

loss1 = 10
loss2 = 10

loss1_t1 = 10
loss2_t1 = 10

loss_terminate = 0.00002
#batch_size = 50

epochs = 3000
batch_size = 100

n = len(trainD)
n_t1 = len(trainD_t1)

pre_value=[]

iter_plot=[]
loss1_plot=[]
iter2_plot=[]
loss2_plot=[]

iter_t1_plot=[]
loss1_t1_plot=[]
iter2_t1_plot=[]
loss2_t1_plot=[]
#Train for large size data
with tf.Session() as sess:
    sess.run(init)
    m = 0
    for epoch in range(epochs):
        if loss1 < loss_terminate:
            exit
        else:

            iter_plot.append(m)
            m +=1
            loss1_plot.append(loss1)

            for iterator in range(0,n, batch_size):
                mini_batch_x = trainD[iterator:iterator+batch_size]
                mini_batch_u = trainL[iterator:iterator+batch_size]
                #print(mini_batch_u)
                sess.run(train_op1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                
                loss1 = sess.run(l1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                if iterator % 1000 == 0:
                    print ("epoch " + str(epoch)+ " iterate " + str(int(iterator/100)) + " loss1: " + str(loss1))

    pre_value = sess.run(pred, {input_0: testD})

UL2loss =np.sum(np.square(pre_value[:,0] - testL[:,0]))
KL2loss =np.sum(np.square(pre_value[:,1] - testL[:,1]))
Ustd =np.std(pre_value[:,0])
Kstd =np.std(pre_value[:,1])

with tf.Session() as sess:
    sess.run(init)
    m = 0
    for epoch in range(epochs):
        if loss2 < loss_terminate:
            exit
        else:

            iter2_plot.append(m)
            m+=1
            loss2_plot.append(loss2)

            for iterator in range(0,n, batch_size):

                mini_batch_x = trainD[iterator:iterator+batch_size]
                mini_batch_u = trainL[iterator:iterator+batch_size]
                #print(mini_batch_u)
                sess.run(train_op2, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                
                loss2 = sess.run(l2, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                if iterator % 1000 == 0:
                    print ("epoch " + str(epoch) + " iterate " + str(int(iterator/100)) + " loss2: " + str(loss2))            
        pre_value2 = sess.run(pred2, {input_0: testD})

UL2loss2 =np.sum(np.square(pre_value2[:,0] - testL[:,0]))
KL2loss2 =np.sum(np.square(pre_value2[:,1] - testL[:,1]))

Ustd2 =np.std(pre_value2[:,0])
Kstd2 =np.std(pre_value2[:,1])

#Train for small size data
with tf.Session() as sess:
    sess.run(init)
    m = 0
    for epoch in range(epochs):
        if loss1_t1 < loss_terminate:
            exit
        else:

            iter_t1_plot.append(m)
            m+=1
            loss1_t1_plot.append(loss1_t1)
            for iterator in range(0,n_t1, batch_size):

                mini_batch_x = trainD_t1[iterator:iterator+batch_size]
                mini_batch_u = trainL_t1[iterator:iterator+batch_size]
                #print(mini_batch_u)
                sess.run(train_op1_t1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                
                loss1_t1 = sess.run(l1_t1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})

                if iterator % 1000 == 0:
                    print ("epoch " + str(epoch)+ " iterate " + str(int(iterator/100)) + " loss1_t1: " + str(loss1_t1))

    pre_value_t1 = sess.run(pred_t1, {input_0: testD_t1})

UL2loss_t1 =np.sum(np.square(pre_value_t1[:,0] - testL_t1[:,0]))
KL2loss_t1 =np.sum(np.square(pre_value_t1[:,1] - testL_t1[:,1]))

Ustd_t1 =np.std(pre_value_t1[:,0])
Kstd_t1 =np.std(pre_value_t1[:,1])

with tf.Session() as sess:
    sess.run(init)
    m = 0 
    for epoch in range(epochs):
        if loss2_t1 < loss_terminate:
            exit
        else:

            iter2_t1_plot.append(m)
            m+=1
            loss2_t1_plot.append(loss2_t1)
            for iterator in range(0,n_t1, batch_size):
 
                mini_batch_x = trainD_t1[iterator:iterator+batch_size]
                mini_batch_u = trainL_t1[iterator:iterator+batch_size]
                #print(mini_batch_u)
                sess.run(train_op2_t1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                
                loss2_t1 = sess.run(l2_t1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                if iterator % 1000 == 0:
                    print ("epoch " + str(epoch)+ " iterate " + str(int(iterator/100)) + " loss2_t1: " + str(loss2_t1))            
    pre_value2_t1 = sess.run(pred2_t1, {input_0: testD_t1})

UL2loss2_t1 =np.sum(np.square(pre_value2_t1[:,0] - testL_t1[:,0]))
KL2loss2_t1 =np.sum(np.square(pre_value2_t1[:,1] - testL_t1[:,1]))

Ustd2_t1 =np.std(pre_value2_t1[:,0])
Kstd2_t1 =np.std(pre_value2_t1[:,1])

x_plot1 = np.arange(-0.5, 0.1, 0.01)
x_plot2 = np.arange(-0.15, 0.1, 0.01)


# f1 = plt.figure()
# ax1 = f1.add_subplot(121)
# ax1.plot(x_plot1,x_plot1, color='black')
# ax1.scatter(pre_value[:,0], testL[:,0],label="Predict u vs True u")
# ax1.legend("No Constraint")
# ax1.set_xlabel('predict data')
# ax1.set_ylabel('test data')
# ax3 = f1.add_subplot(122)
# ax3.plot(x_plot1,x_plot1, color='black')
# ax3.scatter(pre_value2[:,0], testL[:,0],label="Predict u vs True u")
# ax3.legend("With Constraint")
# ax3.set_xlabel('predict data')
# ax3.set_ylabel('test data')
# f2 = plt.figure()
# ax2 = f2.add_subplot(121)
# ax2.plot(x_plot2,x_plot2, color='black')
# ax2.scatter(pre_value[:,1], testL[:,1],label="Predict k vs True k")
# ax2.legend()
# ax2.set_xlabel('predict data')
# ax2.set_ylabel('test data')
# ax4 = f2.add_subplot(122)
# ax4.plot(x_plot2,x_plot2, color='black')
# ax4.scatter(pre_value2[:,1], testL[:,1],label="Predict k vs True k")
# ax4.legend()
# ax4.set_xlabel('predict data')
# ax4.set_ylabel('test data')
#Print L2 loss
print("U and K loss for large size data")
print("UL2loss: " + str(UL2loss))
print("KL2loss: " + str(KL2loss))

print("UL2loss2: " + str(UL2loss2))
print("KL2loss2: " + str(KL2loss2))

print("U and K loss for small size data")
print("UL2loss_t1: " + str(UL2loss_t1))
print("KL2loss_t1: " + str(KL2loss_t1))

print("UL2loss2_t1: " + str(UL2loss2_t1))
print("KL2loss2_t1: " + str(KL2loss2_t1))

#Prit Std
print("U and K std for large size data")
print("Ustd: " + str(Ustd))
print("Kstd: " + str(Kstd))

print("Ustd2: " + str(Ustd2))
print("Kstd2: " + str(Kstd2))

print("U and K loss for small size data")
print("Ustd_t1: " + str(Ustd_t1))
print("Kstd_t1: " + str(Kstd_t1))

print("Ustd2_t1: " + str(Ustd2_t1))
print("Kstd2_t1: " + str(Kstd2_t1))

print(len(testD))
print(len(testD_t1))
#print(iter2_plot)
#Plot of large size data
f1 = plt.figure()
ax1 = f1.add_subplot(111)
plt.title("Predict u vs True u (Large Data)")
ax1.plot(x_plot1,x_plot1, color='black')
ax1.scatter(pre_value[:,0], testL[:,0],label="No Constraint")
ax1.legend()
ax1.set_xlabel('predict data')
ax1.set_ylabel('test data')
ax1_1 = f1.add_subplot(111)
ax1_1.scatter(pre_value2[:,0], testL[:,0],label="With Constraint")
ax1_1.legend()
# ax1_2 = f1.add_subplot(111)
# ax1_2.scatter(pre_value2_t1[:,0], testL[:,0],label="Small data With Constraint")

f2 = plt.figure()
plt.title("Predict k vs True k (Large Data)")
ax2 = f2.add_subplot(111)
ax2.plot(x_plot2,x_plot2, color='black')
ax2.scatter(pre_value[:,1], testL[:,1],label="No Constraint")
ax2.legend()
ax2.set_xlabel('predict data')
ax2.set_ylabel('test data')
ax2 = f2.add_subplot(111)
ax2.scatter(pre_value2[:,1], testL[:,1],label="With Constraint")
ax2.legend()

#Plot of large data loss
f_loss = plt.figure()
ax_loss = f_loss.add_subplot(111)
plt.title("Loss vs Epoch of large size data")
ax_loss.plot(iter_plot[2:-1], loss1_plot[2:-1],label="No Constraint")
ax_loss.legend()
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss2 = f_loss.add_subplot(111)
ax_loss2.plot(iter2_plot[2:-1], loss2_plot[2:-1],label="With Constraint")
ax_loss2.legend()

#Plot of small data loss
f_loss_t1 = plt.figure()
ax_loss_t1 = f_loss_t1.add_subplot(111)
plt.title("Loss vs Epoch of small size data")
ax_loss_t1.plot(iter_t1_plot[2:-1], loss1_t1_plot[2:-1],label="No Constraint")
ax_loss_t1.legend()
ax_loss_t1.set_xlabel('Epoch')
ax_loss_t1.set_ylabel('Loss')
ax_loss2_t1 = f_loss_t1.add_subplot(111)
ax_loss2_t1.plot(iter2_t1_plot[2:-1], loss2_t1_plot[2:-1],label="With Constraint")
ax_loss2_t1.legend()

#Plot of small size data
f3 = plt.figure()
ax3 = f3.add_subplot(111)
plt.title("Predict u vs True u (Small Data)")
ax3.plot(x_plot1,x_plot1, color='black')
ax3.scatter(pre_value_t1[:,0], testL_t1[:,0],label="No Constraint")
ax3.legend()
ax3.set_xlabel('predict data')
ax3.set_ylabel('test data')
ax3 = f3.add_subplot(111)
ax3.scatter(pre_value2_t1[:,0], testL_t1[:,0],label="With Constraint")
ax3.legend()

f4 = plt.figure()
plt.title("Predict k vs True k (Small Data)")
ax4 = f4.add_subplot(111)
ax4.plot(x_plot2,x_plot2, color='black')
ax4.scatter(pre_value_t1[:,1], testL_t1[:,1],label="No Constraint")
ax4.legend()
ax4.set_xlabel('predict data')
ax4.set_ylabel('test data')
ax4 = f4.add_subplot(111)
ax4.scatter(pre_value2_t1[:,1], testL_t1[:,1],label="With Constraint")
ax4.legend()

plt.show()
