import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_node =  a + b

sess = tf.Session()

File_Writer = tf.summary.FileWriter('C:\\Users\\Administrator\\PycharmProjects\\Tensorflow-1\\graph_3', sess.graph)
print(sess.run(add_node,{a:[1,3],b:[5,6]})) # Feed values to place holder

sess.close()
