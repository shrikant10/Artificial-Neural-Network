import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
df = pd.read_csv('air.csv')

y = df['scaled_sound'].as_matrix().reshape(-1,1)
x = df.drop('scaled_sound',axis=1).as_matrix()

x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

x_ = tf.placeholder('float')
y_ = tf.placeholder('float')
keep_prob = tf.placeholder('float')

features = 5
hidden_nodes_1 = 100
hidden_nodes_2 = 10
classes = 1
epoch = 10000
lambda_reg = 0.01

#AdamOptimizer - 1000
# Layers

hidden_layer_1 = {'weights':tf.Variable(tf.random_uniform([features,hidden_nodes_1])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_1]))}

hidden_layer_2 = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_1,hidden_nodes_2])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_2]))}

output_layer = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_2,classes])),
                'biases':tf.Variable(tf.random_uniform([classes]))}

l1 = tf.add(tf.matmul(x_,hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)
l1 = tf.nn.dropout(l1,keep_prob)
l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)
l2 = tf.nn.dropout(l2,keep_prob)
output = tf.add(tf.matmul(l2,output_layer['weights']), output_layer['biases'])
regularisation = 0
#regularisation = lambda_reg*(tf.nn.l2_loss(hidden_layer_1['weights'])+tf.nn.l2_loss(hidden_layer_1['biases'])+ tf.nn.l2_loss(hidden_layer_2['weights'])+ tf.nn.l2_loss(hidden_layer_2['biases'])+ tf.nn.l2_loss(output_layer['weights']) + tf.nn.l2_loss(output_layer['biases']))

error = tf.reduce_mean(tf.square(output-y_)+ regularisation)
#train = tf.train.GradientDescentOptimizer(0.0001).minimize(error)
train = tf.train.AdamOptimizer(0.1).minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        _,e = sess.run([train,error],feed_dict= {x_:x_train,y_:y_train,keep_prob:0.5})
        if i%100==0:
            print('Epoch : ',i+1,'Error : ',e)

    output,Testing_error = sess.run([output,error],feed_dict={x_:x_test,y_:y_test,keep_prob:1.0})

    #--------------------------------Printing the true value and predicted value
    #for i in range(len(y_test)):
        #print(y_test[i],output[i])

    #-------------------------------------Calculating testing error for accuracy
    print('Testing_error : ',Testing_error)

    #-------------------------------------------------------------Plotting data
    plt.plot(range(len(y_test)),y_test,'g',label = 'Actual Data')
    plt.plot(range(len(y_test)),output,'r--',label = 'predicted data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.axis([100, 150,100, 140])
    plt.grid(True)
    plt.show()
