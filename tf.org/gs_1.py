#
# -*- coding:utf-8 -*-
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
print( '<1> 상수 생성')
print(node1, node2)

sess = tf.Session()
print ('<2> 세션 생성')
print(sess.run([node1,node2]))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print ('<3> 플레이스홀더')
print(sess.run(adder_node, {a:3,b:4.5}))
print(sess.run(adder_node, {a:[1,3], b:[2,5]}))

add_and_triple = adder_node * 3.
print ('<4> 노드 연산')
print(sess.run(add_and_triple,{a:3,b:4.5}))

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print('<5> 모델 생성 및 변수')
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print('<6> 모델 평가')
print( sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3] } ) )


fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW,fixb])
print('<7> 모델 수정')
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3] }))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run( train, { x:[1,2,3,4], y: [0,-1,-2,-3] })
    
print( sess.run([W,b]))
