from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imshow
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

weight=np.loadtxt('weights.txt')
const=np.loadtxt('const.txt')
w_assign=tf.assign(W,np.reshape(weight,(784,10)))
b_assign=tf.assign(b,np.reshape(const,(10)))

testim=mnist.test.next_batch(100)

y = tf.matmul(x,W) + b
sess.run([w_assign, b_assign])
result=sess.run([y], {x: testim[0]})

index=tf.argmax(result[0],1).eval()


print("\n\n predicted digit \n"+str(np.reshape(index,(10,10))))

def show_image(testim):
  image=np.reshape(testim[0],(28,28))

  for i in range(1,len(testim[:,1])):
  	curr=np.reshape(testim[i],(28,28))
  	image=np.append(image, curr, axis=1)
 
  n_c=len(image[1,:])
  big_image=image[:,:(n_c/10)]
  for i in range(1,10):
  	s_c=i*n_c/10
  	e_c=(i+1)*n_c/10
  	image_temp=image[:,s_c:e_c]
  	big_image=np.append(big_image,image_temp , axis=0)
  imshow(big_image)
  
show_image(testim[0])  	