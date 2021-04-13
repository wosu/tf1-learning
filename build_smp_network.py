from add_layer import add_layer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot

'''

'''

#使用numpy创建一个矩阵
x_data = np.linspace(-1,1,300)[:,np.newaxis]
print(x_data.shape)
print(x_data)
noises = np.random.normal(0,0.05,size=x_data.shape)
y_data = np.square(x_data) -0.5 + noises

#定义占位符placeholder, xs表示神经网络的输入  ys表示输出
#shape=[None,1]表示输入数据特征只有一种
xs = tf.placeholder(tf.float32,shape=[None,1])
ys = tf.placeholder(tf.float32,shape=[None,1])

#add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
l2 = add_layer(l1,10,5,activation_function=tf.nn.sigmoid)
#add output layer
out = add_layer(l2,5,1,activation_function=None)
#定义loss function,平方损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-out),reduction_indices=[1]))
#优化方法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#以上，只是在构建计算图，没有真正出发计算
#在运行之前，需要对所有的变量进行初始化，调用下面的方式
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

#新建一个画布
fig = plot.figure()
ax = fig.add_subplot(1,1,1)
#散点图
ax.scatter(x_data,y_data)
plot.ion()
plot.show()

#通过session会话，控制训练
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(1000):
        #通过sees.run()启动训练
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 ==0:
            try:
                #清楚上次的轨迹
                ax.lines.remove(lines[0])
            except Exception:
                pass

            #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
            y_pred = sess.run(out,feed_dict={xs:x_data,ys:y_data})
            lines = ax.plot(x_data,y_pred,'r-',lw=2)
            plot.pause(0.2)
