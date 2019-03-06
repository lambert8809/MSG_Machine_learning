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
#layers2 =[1,40, 40, 40, 1]
#layers2 =[1,40, 40, 40, 1]
#layers = [1, 400, 400, 200, 200, 100, 1]

trainD, validateD, testD, trainL, validateL, testL = load_data(1000)

# define variables and functions
weights, biases = initialize_NN(layers)
#weights2, biases2 = initialize_NN(layers2)
input_0 = tf.placeholder(tf.float32,shape=[None,2])
output = tf.placeholder(tf.float32,shape=[None,2])

pred = net_u(input_0,weights,biases)

u = tf.gather(pred, 0, axis=1)
k = pred[:,1]

#u_pred_xx = derivative_u(u,input_0)

u_pred_xx = derivative_u(tf.gather(pred, 0, axis=1),input_0)

u_xx = u_pred_xx[:,0]
#u_pred_x = derivative_u(pred[:,0],input[:,0])

loss_constraint = tf.reduce_sum(tf.square(u_xx - k))
loss_temp = tf.reduce_sum(tf.square(output - pred))
#u_pred2 = (x_u+1)*u_pred+(x_u-1)*u_pred
# Prototype "loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))"

lossL2 = L2Regluarize(weights)*0.002
#+ loss_constraint

#

l1 = tf.reduce_mean(tf.square(output - pred) + lossL2 + 0.0004*loss_constraint )
#l2 =  tf.reduce_mean(tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred))
#loss = tf.reduce_mean(tf.square(m-m_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.002)

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
        if loss1 < 0.002:
            exit
        else:
            rand_index = np.random.choice(len(trainD), size=batch_size)

            #rand_x = np.transpose([x_train[rand_index]])
            #rand_u = np.transpose([u_train[rand_index]])

            rand_x = trainD[rand_index]
            rand_u = trainL[rand_index]

            sess.run(train_op1, feed_dict={input_0: rand_x, output: rand_u})
            
            if step % 100 == 0 or step == 1:
                #loss_temp = sess.run(loss, feed_dict={x_u: x_train, u: u_train, m: m_train})
                loss1 = sess.run(l1, feed_dict={input_0: rand_x, output: rand_u})

                loss_constraint_temp = sess.run(loss_constraint, feed_dict={input_0: rand_x, output: rand_u})
                loss_temp_temp = sess.run(loss_temp, feed_dict={input_0: rand_x, output: rand_u})
                #pred_T_u1_temp = sess.run(pred_T_u1, feed_dict={input_0: rand_x, output: rand_u})

                u_temp = sess.run(u, feed_dict={input_0: rand_x, output: rand_u})
                k_temp = sess.run(k, feed_dict={input_0: rand_x, output: rand_u})
                #pred_temp = sess.run(pred, feed_dict={input_0: rand_x, output: rand_u})
                #u_pred_x_temp = sess.run(u_pred_x, feed_dict={input_0: rand_x, output: rand_u})
                #pred_temp = sess.run(pred, feed_dict={input_0: rand_x, output: rand_u})
                #input_0_T_x_temp =  sess.run(input_0_T_x, feed_dict={input_0: rand_x, output: rand_u})
                u_xx_temp = sess.run(u_xx, feed_dict={input_0: rand_x, output: rand_u})

                print ("Step " + str(step)+ " loss1: " + str(loss1))

                print ("Step " + str(step)+ " loss_constraint_temp: " + str(loss_constraint_temp))
                # print ("Step " + str(step)+ " loss_temp_temp: " + str(loss_temp_temp))
                # #print ("Step " + str(step)+ " pred_temp: " + str(pred_temp))
                # #print ("Step " + str(step)+ " u_temp: " + str(u_temp))
                # print ("Step " + str(step)+ " k_temp: " + str(k_temp))
                # print ("Step " + str(step)+ " u_xx_temp: " + str(u_xx_temp))

                #print(type(pred_T_u))
                #print(tf.DType(input_0))
    pre_value = sess.run(pred, {input_0: testD})

x_plot1 = np.arange(-0.5, 0.1, 0.01)
x_plot2 = np.arange(-0.15, 0.1, 0.01)

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(x_plot1,x_plot1, color='black')
ax1.scatter(pre_value[:,0], testL[:,0],label="Predict u vs True u")
ax1.legend()
ax1.set_xlabel('predict data')
ax1.set_ylabel('test data')
f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(x_plot2,x_plot2, color='black')
ax2.scatter(pre_value[:,1], testL[:,1],label="Predict k vs True k")
ax2.legend()
ax2.set_xlabel('predict data')
ax2.set_ylabel('test data')
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
