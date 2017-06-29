import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('Dow.csv')
df = df.drop(['Date','Adj Close'],axis=1)

#print(df.head(10))
print(df.describe())

y = df['Close'].as_matrix().reshape(-1,1)
x = df.drop('Close',axis=1).as_matrix()

#print(y)

x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)
#print(x)

x_train = x[:201]
y_train = y[:201]

x_test = x[200:]
y_test = y[200:]

x_ = tf.placeholder('float')
y_ = tf.placeholder('float')

features = 4
hidden_nodes_1 = 50
hidden_nodes_2 = 10
classes = 1
epoch = 10000

# Layers

hidden_layer_1 = {'weights':tf.Variable(tf.random_uniform([features,hidden_nodes_1])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_1]))}

hidden_layer_2 = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_1,hidden_nodes_2])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_2]))}

output_layer = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_2,classes])),
                'biases':tf.Variable(tf.random_uniform([classes]))}

l1 = tf.add(tf.matmul(x_,hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)

output = tf.add(tf.matmul(l2,output_layer['weights']), output_layer['biases'])

error = tf.reduce_mean(tf.square(output-y_))
#train = tf.train.AdamOptimizer(0.01).minimize(error)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(error)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        _,e = sess.run([train,error],feed_dict= {x_:x_train,y_:y_train})
        print('Epoch : ',i+1,'Error : ',e)

    output,cost = sess.run([output,error],feed_dict={x_:x_test,y_:y_test})
    print(cost)
    for i in range(len(y_test)):
        print(y_test[i],output[i])


    plt.plot(range(len(y_test)),y_test,'r')
    plt.plot(range(len(y_test)),output)
    plt.xlabel('day')
    plt.ylabel('Closing price')
    plt.show()
