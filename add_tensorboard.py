'''
tf.name_scope,给变量加前缀，相当于分类管理，模块化，
    参考：https://www.zhihu.com/question/54513728
'''

from add_layer import add_layer
import tensorflow as tf

#define placeholder for network
with tf.name_scope('inputs'):
    #n行1列
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

#hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
l2 = add_layer(l1,10,5,activation_function=tf.nn.relu)

pred = add_layer(l2,5,1,activation_function=None)

#defina loss function
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred),reduction_indices=[1]))

#define train
with tf.name_scope('train'):
    tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    # tf.train.SummaryWriter soon be deprecated, use following
    if int((tf.__version__).split('.')[1]) < 12 and int(
            (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
        writer = tf.train.SummaryWriter('logs/', sess.graph)
    else:
        # tensorflow version >= 0.12
        writer = tf.summary.FileWriter("logs/", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

# 启动 tensorboard --logdir=logs
# http://LAPTOP-3GA6E8CT:6006/
#在启动tensorboard时会遇到拒绝访问问题，参考：https://blog.csdn.net/weixin_41887832/article/details/88912106
# $ tensorboard --logdir=logs
