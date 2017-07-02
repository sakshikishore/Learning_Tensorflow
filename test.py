from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imshow
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 5])
W = tf.Variable(tf.zeros([5,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b

sess.run(tf.global_variables_initializer())

result=sess.run([y], {x: np.reshape([1,2,3,4,5],(1,5))})
print("Result=",result)

index=tf.argmax(result[0],1).eval()

print(index)

