import tensorflow as tf

x_train = [[0,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,0,1,1],
            [0,1,0,0],
            [0,1,0,1],
            [0,1,1,0],
            [0,1,1,1],
            [1,0,0,0],
            [1,0,0,1],
            [1,0,1,0],
            [1,0,1,1],
            [1,1,0,0],
            [1,1,0,1],
            [1,1,1,0],
            [1,1,1,1]]

y_train = [[0,1],
            [1,0],[1,0],[0,1],[1,0],
            [0,1],[0,1],[1,0],[1,0],
            [0,1],[0,1],[1,0],[0,1],
            [1,0],[1,0],[0,1]]

x = tf.placeholder("float")
y = tf.placeholder("float")

hidden_nodes_1 = 10
hidden_nodes_2 = 5
features = 4

# Hidden Layer

hidden_layer_1 = {'weights':tf.Variable(tf.random_uniform([features,hidden_nodes_1])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_1]))}

hidden_layer_2 = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_1,hidden_nodes_2])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes_2]))}

output_layer = {'weights':tf.Variable(tf.random_uniform([hidden_nodes_2,2])),
                'biases':tf.Variable(tf.random_uniform([2]))}

l1 = tf.add(tf.matmul(x,hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)

output_layer = tf.add(tf.matmul(l2,output_layer['weights']), output_layer['biases'])
output = tf.nn.softmax(output_layer)

error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels = y))
train = tf.train.GradientDescentOptimizer(0.1).minimize(error)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _, e = sess.run([train,error],feed_dict= {x:x_train, y:y_train})
        if i%100==0:
            print('Epoch :',i,'Error: ',e)

    pred = tf.argmax(output,1)
    print(pred.eval(feed_dict={x:[[0,0,1,1]]}))
