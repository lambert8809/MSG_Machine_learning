import tensorflow as tf
import numpy as np
import pandas as pd
from pyDOE import lhs
import time
import matplotlib.pyplot as plt

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

#layers = [1, 5, 5, 5,1]
layers = [1, 50, 1] 
#layers2 =[1,40, 40, 40, 1]
layers2 =[1,40, 40, 40, 1]
#layers = [1, 400, 400, 200, 200, 100, 1]
data = pd.read_csv('Data2.csv')
x = data['x'].values
u = data['u_mm'].values
m = data['M'].values
# Scale input and output for better training
#x = (x - 5)/5
# scale deflection for better training
u = u * 100
m = m
# Take training and testing data from the database 
n=2
x_train0 = x[::n]
u_train0 = u[::n]
m_train0 = m[::n]
x_test = x[1::10]
u_test = u[1::10]
m_test = m[1::10]
# permutation the samples to get more uniform training
newindex = np.random.permutation(len(x_train0))
x_train = x_train0[newindex[:]]
u_train = u_train0[newindex[:]]
m_train = m_train0[newindex[:]]
# reshape the data array
x_train = x_train.reshape(len(x_train),1)
u_train = u_train.reshape(len(u_train),1)
m_train = m_train.reshape(len(m_train),1)
x_test = x_test.reshape(len(x_test),1)
u_test = u_test.reshape(len(u_test),1)
m_test = m_test.reshape(len(m_test),1)
# define variables and functions
weights, biases = initialize_NN(layers)
weights2, biases2 = initialize_NN(layers2)
x_u = tf.placeholder(tf.float32,shape=[None,1])       
u = tf.placeholder(tf.float32,shape=[None,1])

m = tf.placeholder(tf.float32,shape=[None,1])  

u_pred = net_u(x_u,weights,biases)
u_x = derivative_u_x(u_pred,x_u)

u_xx = derivative_u(u_pred,x_u)

m_pred = net_u(u_xx,weights2,biases2)

M_xx = derivative_M(m_pred,x_u)
M_x = derivative_M1(m_pred,x_u)
#u_pred2 = (x_u+1)*u_pred+(x_u-1)*u_pred
# Prototype "loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))"
l1 = tf.reduce_mean(tf.square(u - u_pred))
l2 =  tf.reduce_mean(tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred)+tf.square(m-m_pred))
#loss = tf.reduce_mean(tf.square(u - u_pred))
#loss = tf.reduce_mean(tf.square(m-m_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train_op1 = optimizer.minimize(l1)
train_op2 = optimizer.minimize(l2)
#train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

num_steps = 1000
loss1 = 10
batch_size = 200

pre_value=[]
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps*40+1):
        if loss1 < 0.01:
            exit
        else:
            rand_index = np.random.choice(len(x_train), size=batch_size)

            #rand_x = np.transpose([x_train[rand_index]])
            #rand_u = np.transpose([u_train[rand_index]])

            rand_x = x_train[rand_index]
            rand_u = u_train[rand_index]

            sess.run(train_op1, feed_dict={x_u: rand_x, u: rand_u, m: m_train})
            
            if step % 100 == 0 or step == 1:
                #loss_temp = sess.run(loss, feed_dict={x_u: x_train, u: u_train, m: m_train})
                loss1 = sess.run(l1, feed_dict={x_u: x_train, u: u_train, m: m_train})

                #w_sum = sess.run(tf.reduce_mean(weights[step]))

                #print ("Step " + str(step)+ ": " + str(loss_temp))
                print ("Step " + str(step)+ " loss1: " + str(loss1))
                #print ("Step " + str(step)+ " rand_x: " + str(rand_x))
    pre_value = sess.run(u_pred, {x_u: x_test})
    pre_value_test = sess.run(u_pred, {x_u: x_train})
    pre_value_u_xx = sess.run(u_xx, {x_u: x_test})
    pre_value_u_x = sess.run(u_x, {x_u: x_test})
# with tf.Session() as sess:
#     sess.run(init)
#     for step in range(1, num_steps+1):
#         sess.run(train_op2, feed_dict={x_u: x_train, u: u_train, m: m_train})
#         if step % 100 == 0 or step == 1:
#             #loss_temp = sess.run(loss, feed_dict={x_u: x_train, u: u_train, m: m_train})
#             #loss1 = sess.run(l1, feed_dict={x_u: x_train, u: u_train, m: m_train})
#             loss2 = sess.run(l2, feed_dict={x_u: x_train, u: u_train, m: m_train})
#             # print ("Step " + str(step)+ ": " + str(loss_temp))
#             #print ("Step " + str(step)+ " loss1: " + str(loss1))
#             print ("Step " + str(step)+ " loss2: " + str(loss2))
#     #pre_value = sess.run(u_pred, {x_u: x_test})
#     pre_value2 = sess.run(m_pred, {x_u: x_test})
#     pre_value3 = sess.run(M_xx, {x_u: x_test}) 
#     pre_value4 = sess.run(M_x, {x_u: x_test}) 
#     print ("Step " + str(step)+ " pre_value3: " + str(pre_value3))
#     print ("Step " + str(step)+ " pre_value3: " + str(pre_value4))

#print (pre_value.shape)
#write_to_file(x_train0, pre_value, u_train)
f1 = plt.figure()
# f2 = plt.figure()
# f3 = plt.figure()
# f4 = plt.figure()
f5 = plt.figure()
f6 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.plot(x_test[:], pre_value[:], 'r-')
ax1.plot(x_test[:], u_test[:], 'b+')
# ax2 = f2.add_subplot(111)
# ax2.plot(x_test[:], pre_value2[:], 'r-')
# ax2.plot(x_test[:], m_test[:], 'b+')
# ax3 = f3.add_subplot(111)
# ax3.plot(x_test[:], pre_value3[:], 'r-')
# ax4 = f4.add_subplot(111)
# ax4.plot(x_test[:], pre_value4[:], 'r-')
ax5 = f5.add_subplot(111)
ax5.plot(x_test[:], pre_value_u_xx[:], 'r-')
ax6 = f6.add_subplot(111)
ax6.plot(x_test[:], pre_value_u_x[:], 'r-')
plt.show()

'''
f1 = plt.figure(1)
plt.plot(x_test[:], pre_value[:], 'r-')
plt.plot(x_test[:], u_test[:], 'b+')
f1.show()

f2 = plt.figure(2)
plt.plot(x_test[:], pre_value2[:], 'r-')
plt.plot(x_test[:], m_test[:], 'b+')
f2.show()
'''