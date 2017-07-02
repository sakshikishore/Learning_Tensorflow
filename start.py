import tensorflow as tf

#declaring constant

n1=tf.constant(1.0, dtype=tf.float32)
n2=tf.constant(2.0, dtype=tf.float32)
print(n1, n2)      #will print their type, however on running the session one can get values

sess=tf.Session()

x=sess.run([n1,n2])                 #values are assigned while running
print(x)

n3=tf.subtract(n2,n1)

sess=tf.Session()

print(sess.run(n3))


#######################
# Declaring placeholder (promising to provide value later)
#######################

p1=tf.placeholder(tf.float32)
p2=tf.placeholder(tf.float32)

sum=p1+p2      
#or sum=tf.add(v1, v2)

# sess=tf.Session()
print(sess.run(sum, {p1:1, p2:2}))
print(sess.run(sum, {p1:[1,2,3], p2:[1,1,1]}))


######################
# Variables
######################

v1=tf.Variable([0.3], dtype=tf.float32)
v2=tf.Variable([-0.1], dtype=tf.float32)
init = tf.global_variables_initializer()        

sess.run(init)                             #to initialize all the variable you must explicitly call this
print(sess.run(v1+v2))

ass_v1=tf.assign(v1, [1])
ass_v2=tf.assign(v2, [2])

sess.run([ass_v1, ass_v2])
print(sess.run(v1+v2))

############################################################################################
# Basically everything is an operation and you must put all that in sess.run() 
# *only takes single argument use list*  to execute.
############################################################################################
