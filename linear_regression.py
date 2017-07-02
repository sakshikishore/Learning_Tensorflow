import tensorflow as tf
x_train=tf.placeholder(tf.float32)
y_train=tf.placeholder(tf.float32)

W= tf.Variable([30], dtype=tf.float32)
b=tf.Variable([-3 ], dtype=tf.float32)

lin_func=W*x_train+b

cost_arr=tf.square(lin_func- y_train)
total_cost=tf.reduce_sum(cost_arr)

optimizer=tf.train.GradientDescentOptimizer(0.01)          #giving alpha value for gradient descent
train=optimizer.minimize(total_cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 

for i in range(1000):
	sess.run(train, {x_train: [1,2,3,4], y_train: [0, -1, -2, -3]})
	curr_W, curr_b, curr_loss = sess.run([W, b, total_cost], {x_train: [1,2,3,4], y_train: [0, -1, -2, -3]})
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))



