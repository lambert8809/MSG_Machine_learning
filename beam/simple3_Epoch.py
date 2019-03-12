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
        W = tf.Variable(tf.random_normal([layers[l], layers[l+1]], dtype=tf.float32), dtype=tf.float32)
        b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
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
layers = [2, 100,100,100, 2] 
layers2 =[2, 100,100,100, 2] 
#layers2 =[1,40, 40, 40, 1]
#layers = [1, 400, 400, 200, 200, 100, 1]

trainD, validateD, testD, trainL, validateL, testL = load_data(10000)

# define variables and functions
weights, biases = initialize_NN(layers)
weights2, biases2 = initialize_NN(layers2)

input_0 = tf.placeholder(tf.float32,shape=[None,2])
output = tf.placeholder(tf.float32,shape=[None,2])

pred = net_u(input_0,weights,biases)

pred2 = net_u(input_0,weights2,biases2)


u_pred_xx = derivative_u(tf.gather(pred2, 0, axis=1),input_0)

u = tf.gather(pred2, 0, axis=1)
k = pred2[:,1]

u_xx = u_pred_xx[:,0]


loss_constraint = tf.reduce_sum(tf.square(u_xx - k))

#u_pred2 = (x_u+1)*u_pred+(x_u-1)*u_pred
# Prototype "loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))"

lossL2 = L2Regluarize(weights)*0.002
lossL2_2 = L2Regluarize(weights2)*0.002
#+ 0.0002*loss_constraint

l1 = tf.reduce_mean(tf.square(output - pred) + lossL2)
l2 = tf.reduce_mean(tf.square(output - pred2) + lossL2_2+ 0.0002*loss_constraint)

#l2 =  tf.reduce_mean(tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred))
#loss = tf.reduce_mean(tf.square(m-m_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.002)

train_op1 = optimizer.minimize(l1)
train_op2 = optimizer.minimize(l2)
#train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

num_steps = 1000
loss1 = 10
loss2 = 10
loss_terminate = 2
#batch_size = 50

epochs = 300
batch_size = 100
n = len(trainD)

pre_value=[]
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for k in range(0,n, batch_size):
            if loss1 < loss_terminate:
                exit
            else:
                mini_batch_x = trainD[k:k+batch_size]
                mini_batch_u = trainL[k:k+batch_size]
                #print(mini_batch_u)
                sess.run(train_op1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                
                if k*batch_size % 100 == 0 or k*batch_size == 1:
                    loss1 = sess.run(l1, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                    print ("epoch " + str(epoch)+ " loss1: " + str(loss1))

        pre_value = sess.run(pred, {input_0: testD})
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for k in range(0,n, batch_size):
            if loss2 < loss_terminate:
                exit
            else:
                mini_batch_x = trainD[k:k+batch_size]
                mini_batch_u = trainL[k:k+batch_size]
                #print(mini_batch_u)
                sess.run(train_op2, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                
                if k*batch_size % 100 == 0 or k*batch_size == 1:
                    loss2 = sess.run(l2, feed_dict={input_0: mini_batch_x, output: mini_batch_u})
                    print ("epoch " + str(epoch)+ " loss2: " + str(loss2))            
        pre_value2 = sess.run(pred2, {input_0: testD})

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
f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(x_plot1,x_plot1, color='black')
ax1.scatter(pre_value[:,0], testL[:,0],label="Predict u vs True u")
ax1.legend("No Constraint")
ax1.set_xlabel('predict data')
ax1.set_ylabel('test data')
ax3 = f1.add_subplot(111)
ax3.plot(x_plot1,x_plot1, color='orange')
ax3.scatter(pre_value2[:,0], testL[:,0],label="Predict u vs True u")
ax3.legend("With Constraint")
ax3.set_xlabel('predict data')
ax3.set_ylabel('test data')
f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(x_plot2,x_plot2, color='black')
ax2.scatter(pre_value[:,1], testL[:,1],label="Predict k vs True k")
ax2.legend()
ax2.set_xlabel('predict data')
ax2.set_ylabel('test data')
ax4 = f2.add_subplot(111)
ax4.plot(x_plot2,x_plot2, color='orange')
ax4.scatter(pre_value2[:,1], testL[:,1],label="Predict k vs True k")
ax4.legend()
ax4.set_xlabel('predict data')
ax4.set_ylabel('test data')
plt.show()
