import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Read Dataset

def read_dataset():
    df = pd.read_csv("H:\\Machine Learning\\sonar_all_data.csv")
    # print(df.head())

    X = df[df.columns[0:60]].values  # from 0th col. till 59th col is our features
    y1 = df[df.columns[60]]  # final 60th col is our Label

    # Encode the dependent variable i.e Label

    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X, Y, y1)


# we will use one hot encoding - which is only one is input is active at a time
# Define Encoder function

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Read dataset

X, Y, y1 = read_dataset()  # X is feature and Y is one hot encoded label

# We need to shuffle dataset as it is by default in order in CSV
#X, Y = shuffle(X, Y, random_state=1)

# Convert dataset in train and test part

#train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# Inspect the shape of training and testing set
#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)

model_path = "C:/Users/Administrator/PycharmProjects/Tensorflow-1/NMI"

# Define important parameters and variables to work with tensors
learning_rate = 0.3
training_epochs = 1000  # total no of iterations defined in order to minimize the error
cost_history = np.empty(shape=[1], dtype=float)  # loss function
n_dim = 60  # n_dim - shape of your features which is stored in X , that too it only include no of columns hence [1]
n_class = 2  # since we have only 2 classes
#print("n_dim", n_dim)

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
# X is a placeholder whe we gona feed in our dataset and shape of X can be any value as defied as NONE
W = tf.Variable(tf.zeros([n_dim, n_class]))  # veriable W which is initialze by zeros and have shape n_dim and n_class which is 2
b = tf.Variable(tf.zeros([n_class]))  # vaiable b which was initialize by zeros and have shape n_class which is 2
y_ = tf.placeholder(tf.float32,[None, n_class])  # y_ is output of our model shape is none - any value and n_class which is 2


# define the model

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with ReLU activation
    # first hidden layer in which matrix multiplication of input layer x and the weight and then we are add it to the biases
    # then we are going to pass it through sigmoid activation function

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # 2nd Hidden layer with sigmoid activation
    # input of 2nd hidden layer is out of first hidden layer
    # then again matrixmultiplication of layer_1 with weights and add it to the biases

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # 3rd Hidden layer with sigmoid activation

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # 4rth hidden layer with ReLU activation function

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # output layer with linear activation

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# define Weights and biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# inintailize all the variables

init = tf.global_variables_initializer()

# create saver object in order to save our model
saver = tf.train.Saver()

# call our model
y = multilayer_perceptron(x, weights, biases)

# define cost function/ loss function and optimizers
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))  # (model o/p , actual o/p)
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()  # create session object to launch the graph
sess.run(init)  # this will initialize all the variables
saver.restore(sess, model_path) # We need to call restore function in order to restore the model which we have

# calculate cost and accuracy of each epoch

mse_history = []
accuracy_history = []

prediction = tf.argmax(y, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(" 0 stands for M - MINE and 1 stands for R - ROCK")

for i in range (93, 101):
    prediction_run = sess.run(prediction, feed_dict={x: X[i].reshape(1,60)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].(1,60), y_: y1[i].reshape(1,60)})
    print("Original class :", y1[i], "Predicted values  :",prediction_run, "Accuracy :",accuracy_run)





#for epoch in range(training_epochs):
    # first we gona run with training step and we gona feed with the value of training data
#sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    # run cost/loss funtion and aging feed the values of train_x and train_y
    #cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    #cost_history = np.append(cost_history,cost)  # cost histroy will append one by one for every epoch by cost from above
    #correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))  # difference between actual output and the model output

    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #pred_y = sess.run(y, feed_dict={x: test_x})  # actual output-y feed in a test data in order to see how accurate the model is

    #mse = tf.reduce_mean(tf.square(pred_y - test_y))  # diff bet predicted and test data

    #mse_ = sess.run(mse)  # launch graph
    #mse_history.append(mse_)  # keep on updating mse vale for each epoch

    #accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))  # finding accuracy on train data

    #accuracy_history.append(accuracy)  # append accuracy on every epoch

    #print('epoch : ', epoch, '-', 'cost : ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

#save_path = saver.save(sess, model_path)
#print("Model saved in file : %s" % save_path)

# plot mse and accuracy graph

#plt.plot(mse_history, 'r')
#plt.show()
#plt.plot(accuracy_history)
#plt.show()

# print the final accuracy


#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print("Test Accuracy : ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))

# print final MSE


#pred_y = sess.run(y, feed_dict={x: test_x})
#mse = tf.reduce_mean(tf.square(pred_y - test_y))
#print("Mse : %.4f" % sess.run(mse))
