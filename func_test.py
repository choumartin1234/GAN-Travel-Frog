import tensorflow as tf

a=tf.Variable(initial_value=[1,2,3,4])
c=[1,2,3,4]
b=reversed(c[:-1])
print([_ for _ in b])