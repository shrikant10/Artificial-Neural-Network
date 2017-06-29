import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pd.read_csv('Cars.csv')
#df.keys()
#df.head()

var_mod = ['buying','maint','doors','persons','lug_boot','safety','class_value']

le = LabelEncoder()
enc = OneHotEncoder()

#-----------------------------------Labeling the data and converting it into int
for i in var_mod:
    df[i] = le.fit_transform(df[i])

x = df.drop('class_value',axis=1).as_matrix()
#print(x)

#-----------------------------------Converting y to one_hot
#print(df['class_value'].as_matrix().reshape(-1,1))
y = enc.fit_transform(df['class_value'].as_matrix().reshape(-1,1)).toarray()
#print(y)

#-----------------------------------Normailisation is not required
#x -= np.mean(x, axis=0)
#x /= np.std(x, axis=0)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#print(x_train,y_train)

x_ = tf.placeholder('float')
y_ = tf.placeholder('float')
keep_prob = tf.placeholder('float')

features = 6
hidden_nodes_1 = 100
hidden_nodes_2 = 50
classes = 4
epoch = 1000
learn_rate = 0.01
lambda_reg = 0.001  #Regularisation constant works best at 0.001 on increasing results poor

#------------------------------------------------------------------------Layers

hidden_layer_1 = {'weights':tf.Variable(tf.random_uniform([features,hidden_nodes_1])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_1]))}

hidden_layer_2 = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_1,hidden_nodes_2])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_2]))}

output_layer = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_2,classes])),
                'biases':tf.Variable(tf.random_uniform([classes]))}

#-----------------------------------------------------------Applying y = W*x + B
l1 = tf.add(tf.matmul(x_,hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)
l1 = tf.nn.dropout(l1,keep_prob)

l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)
l1 = tf.nn.dropout(l2,keep_prob)

output = tf.add(tf.matmul(l2,output_layer['weights']), output_layer['biases'])
#output = tf.nn.softmax(output)


#-----------------------------------------------------------------Regularisation
regularisation = 0
#regularisation = lambda_reg*(tf.nn.l2_loss(hidden_layer_1['weights'])+tf.nn.l2_loss(hidden_layer_1['biases'])+ tf.nn.l2_loss(hidden_layer_2['weights'])+ tf.nn.l2_loss(hidden_layer_2['biases'])+ tf.nn.l2_loss(output_layer['weights']) + tf.nn.l2_loss(output_layer['biases']))

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels = y_) + regularisation)
#train = tf.train.AdamOptimizer(learn_rate).minimize(error)
train = tf.train.AdamOptimizer(learn_rate).minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        _,e = sess.run([train,error],feed_dict= {x_:x_train,y_:y_train,keep_prob:0.5})
        #print('Epoch : ',i+1,'Error : ',e)


    #                                  _Converting y_test from one_hot to normal
    decoded = tf.argmax(y_test,axis = 1)
    print('Actual Classes :')
    print(sess.run(decoded))

    #                                  _Predicting our model on testing data
    prediction = tf.argmax(output,1)
    output_array = prediction.eval(feed_dict= {x_:x_test,keep_prob:1.0})
    print('Predicted Classes :')
    print(output_array)


    #                                  _Counting the correctly classified items
    count = 0
    for i in range(len(y_test)):
        if output_array[i]==sess.run(decoded[i]):
            count+=1
    print(count,' Correct out of ',len(y_test))


    #                                  _Calculating the accuracy of our system
    correct = tf.equal(tf.argmax(output,1),tf.argmax(y_test,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print('Accuracy :',accuracy.eval({x_:x_test,y_:y_test,keep_prob:1.0}))
