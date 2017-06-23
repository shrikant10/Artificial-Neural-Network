import tensorflow as tf

x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[1,0],[0,1],[0,1],[1,0]]

x = tf.placeholder("float")
y = tf.placeholder("float")

hidden_nodes = 20

# Hidden Layer

hidden_layer = {'weights':tf.Variable(tf.random_uniform([2,hidden_nodes])),
                'biases':tf.Variable(tf.random_uniform([hidden_nodes]))}

output_layer = {'weights':tf.Variable(tf.random_uniform([hidden_nodes,2])),
                'biases':tf.Variable(tf.random_uniform([2]))}

l1 = tf.add(tf.matmul(x,hidden_layer['weights']), hidden_layer['biases'])
l1 = tf.nn.relu(l1)

output_layer = tf.add(tf.matmul(l1,output_layer['weights']), output_layer['biases'])
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
    print(pred.eval(feed_dict={x:[[1,1]]}))
