import tensorflow as tf


# Model parameters
m = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Inputs and outputs

x = tf.placeholder(tf.float32)

linear_model = m * x + b

y = tf.placeholder(tf.float32)

# Loss function

square_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(square_delta)

# initialize all the variables ,
# global variable initializer will add an operation to initialize the variable

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

