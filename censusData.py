"""
Project 1

At the end you should see something like this
Step Count:1000
Training accuracy: 0.8999999761581421 loss: 0.42281264066696167
Test accuracy: 0.8199999928474426 loss: 0.4739704430103302

play around with your model to try and get an even better score
"""

import tensorflow as tf
import dataUtils

training_data, training_labels = dataUtils.readData("project1trainingdata.csv")
test_data, test_labels = dataUtils.readData("project1testdata.csv")


# Build tensorflow blueprint
## Tensorflow placeholder
input_placeholder = tf.placeholder(tf.float32, shape=[None, 113])
## Neural network hidden layers
# layer 1
weight1 = tf.get_variable("weight1", shape=[113, 150], initializer=tf.contrib.layers.xavier_initializer())
bias1 = tf.get_variable("bias1", shape=[150], initializer=tf.contrib.layers.xavier_initializer())
hidden_layer_1 = tf.nn.relu(tf.matmul(input_placeholder, weight1) + bias1)

# layer 2
weight2 = tf.get_variable("weight2", shape=[150, 125], initializer=tf.contrib.layers.xavier_initializer())
bias2 = tf.get_variable("bias2", shape=[125], initializer=tf.contrib.layers.xavier_initializer())
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weight2) + bias2)

# layer 3
hidden_layer_3 = tf.layers.dense(hidden_layer_2, 100, activation=tf.nn.relu)

## Logit layer
logits = tf.nn.softmax(tf.layers.dense(hidden_layer_3, 2, activation=None))

## label placeholder
label_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

## loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_placeholder, logits=logits))
## backpropagation algorithm
train = tf.train.AdamOptimizer().minimize(loss)

accuracy = dataUtils.accuracy(logits, label_placeholder)



## Make tensorflow session
with tf.Session() as sess:
    ## Initialize variables
    sess.run(tf.global_variables_initializer())


    step_count = 0
    while True:
        step_count += 1

        batch_training_data, batch_training_labels = dataUtils.getBatch(data=training_data, labels=training_labels, batch_size=100)

        training_accuracy, training_loss, logits_output, _ = \
            sess.run([accuracy, loss, logits, train],
                        feed_dict={input_placeholder: batch_training_data,
                                   label_placeholder: batch_training_labels})

        # every 10 steps check accuracy
        if step_count % 10 == 0:
            batch_test_data, batch_test_labels = dataUtils.getBatch(data=test_data, labels=test_labels,
                                                                            batch_size=1000)
            test_accuracy, test_loss, logits_output =\
                sess.run([accuracy, loss, logits],
                         feed_dict={input_placeholder: batch_test_data,
                                    label_placeholder: batch_test_labels})

            #print("Logits {}".format(logits_output))
            print("Step Count:{}".format(step_count))
            print("Training accuracy: {} loss:{}".format(training_accuracy, training_loss))
            print("Test accuracy: {} loss: {}".format(test_accuracy, test_loss))


        # stop training after 1,000 steps
        if step_count > 1000:
            break
