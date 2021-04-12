import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    #参数矩阵，定义参数矩阵
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #定义bias,一般初始化为0
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    #z=wx + b矩阵相乘
    z = tf.matmul(inputs,Weights) + biases
    #如果有激活函数，则z输入到激活函数
    if activation_function is None:
        return z
    else:
        return activation_function(z)


