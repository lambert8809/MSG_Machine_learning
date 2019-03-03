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
    us = us/100/1000
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
    M_x = tf.gradients(us, xs)[0]
    return M_x

def L2Regluarize(weights):
    L2_temp = 0
    for i in range(len(weights)):
        #print("L2" + str(weights[i]))
        L2_reg = L2_temp + tf.nn.l2_loss(weights[i])
    return L2_reg

#layers = [1, 5, 5, 5,1]
layers = [2, 40, 2] 
#layers2 =[1,40, 40, 40, 1]
#layers2 =[1,40, 40, 40, 1]
#layers = [1, 400, 400, 200, 200, 100, 1]

trainD, validateD, testD, trainL, validateL, testL = load_data(1000)

# define variables and functions
weights, biases = initialize_NN(layers)
#weights2, biases2 = initialize_NN(layers2)
input = tf.placeholder(tf.float32,shape=[None,2])
output = tf.placeholder(tf.float32,shape=[None,2])

pred = net_u(input,weights,biases)

#u_pred2 = (x_u+1)*u_pred+(x_u-1)*u_pred
# Prototype "loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))"

lossL2 = L2Regluarize(weights)*0.002

l1 = tf.reduce_mean(tf.square(output - pred) + lossL2)
#l2 =  tf.reduce_mean(tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred))
#loss = tf.reduce_mean(tf.square(m-m_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.02)

train_op1 = optimizer.minimize(l1)
#train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

num_steps = 1000
loss1 = 10
batch_size = 50

pre_value=[]
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps*40+1):
        if loss1 < 0.0002:
            exit
        else:
            rand_index = np.random.choice(len(trainD), size=batch_size)

            #rand_x = np.transpose([x_train[rand_index]])
            #rand_u = np.transpose([u_train[rand_index]])

            rand_x = trainD[rand_index]
            rand_u = trainL[rand_index]

            sess.run(train_op1, feed_dict={input: rand_x, output: rand_u})
            
            if step % 100 == 0 or step == 1:
                #loss_temp = sess.run(loss, feed_dict={x_u: x_train, u: u_train, m: m_train})
                loss1 = sess.run(l1, feed_dict={input: rand_x, output: rand_u})

                #w_sum = sess.run(tf.reduce_mean(weights[step]))

                #print ("Step " + str(step)+ ": " + str(loss_temp))
                print ("Step " + str(step)+ " loss1: " + str(loss1))
                #print ("Step " + str(step)+ " rand_x: " + str(rand_x))
    pre_value = sess.run(pred, {input: testD})

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.scatter(pre_value[:,0], testL[:,0],label="Predict u vs True u")
ax1.legend()
f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.scatter(pre_value[:,1], testL[:,1],label="Predict k vs True k")
ax2.legend()
#f1 = plt.figure()
# f2 = plt.figure()
# f3 = plt.figure()
# f4 = plt.figure()
#f5 = plt.figure()
#f6 = plt.figure()
#ax1 = f1.add_subplot(111)
#ax1.set_title('Comparison')
#ax1.plot(x_test[:], pre_value[:], 'r-')
#ax1.plot(x_test[:], u_test[:], 'b+')
# ax2 = f2.add_subplot(111)
# ax2.plot(x_test[:], pre_value2[:], 'r-')
# ax2.plot(x_test[:], m_test[:], 'b+')
# ax3 = f3.add_subplot(111)
# ax3.plot(x_test[:], pre_value3[:], 'r-')
# ax4 = f4.add_subplot(111)
# ax4.plot(x_test[:], pre_value4[:], 'r-')
#ax6 = f6.add_subplot(111)
#ax6.set_title('1st Order Differentiation')
#ax6.plot(x_test[:], pre_value_u_x[:], 'r-')
#ax5 = f5.add_subplot(111)
#ax5.set_title('2nd Order Differentiation')
#ax5.plot(x_test[:], pre_value_u_xx[:], 'r-')
plt.show()
