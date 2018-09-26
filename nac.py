import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

var_init = lambda x: tf.variance_scaling_initializer(scale=x, distribution='uniform')

class NACCell(object):

    def __init__(self, in_dim, out_dim):

        with tf.variable_scope("nac-cell"):
            self.w = tf.get_variable(name = 'w', shape = [in_dim, out_dim], initializer=var_init(1.0))
            self.m = tf.get_variable(name = 'm', shape = [in_dim, out_dim], initializer=var_init(1.0))
            self.b = tf.get_variable(name = 'b', shape = [1, out_dim])

    def __call__(self, x):

        w = tf.multiply(tf.tanh(self.w), tf.sigmoid(self.m))
        y = tf.matmul(x, w) + self.b

        return y


class NAC(object):

    def __init__(self, in_dim=2, out_dim=1, hidden_dim=2, n_stacks=2):

        self.layers = []

        with tf.variable_scope("nac"):

            for i in range(n_stacks):
                    self.layers.append(
                        NACCell(
                            in_dim if i == 0 else hidden_dim, 
                            out_dim if i == n_stacks-1 else hidden_dim
                        )
                    )

    def __call__(self, x):

        y = x

        for layer in self.layers:
            y = layer(y)

        return y

class NALUCell(object):

    def __init__(self, in_dim, out_dim, eps=10**-5):
        
        with tf.variable_scope("nalu-cell"):
            self.add_cell = NACCell(in_dim, out_dim)
            self.mul_cell = NACCell(in_dim, out_dim)
            self.G = tf.get_variable('g', shape = [in_dim, out_dim], initializer=var_init(1/3))
            self.eps = eps

    def __call__(self, x):

        a = self.add_cell(x)
        m = self.mul_cell(tf.log(tf.abs(x) + self.eps))
        g = tf.sigmoid(tf.matmul(x, self.G))
        y = tf.multiply(g, a) + tf.multiply(1-g, m)

        return y

class NALU(object):

    def __init__(self, in_dim, out_dim, hidden_dim, n_stacks, eps):

        self.layers = []

        with tf.variable_scope("nalu"):
            for i in range(n_stacks):
                self.layers.append(
                    NALUCell(
                        in_dim if i == 0 else hidden_dim,
                        out_dim if i == n_stacks-1 else hidden_dim
                    )
                )

    def __call__(self, x):

        y = x

        for layer in self.layers:
            y = layer(y)

        return y


def loss_fn(nac, x, target_y):
    y = nac(x)
    return tf.reduce_mean(tf.pow(target_y - y, 2)) 

# add_op = lambda x, y: x + y
mul_op = lambda x, y: x*y
add_op = mul_op


if __name__ == "__main__":

     with tf.device("/gpu:0"):
        optim = tf.train.AdamOptimizer(0.01)

        x1 = tf.random_uniform([100, 1], 1, 10)
        x2 = tf.random_uniform([100, 1], 1, 10)
        target_y = add_op(x1, x2)
        x = tf.concat([x1, x2], axis=1)

        # nac = NAC()
        # nac = NACCell(2, 1)
        # nac = NALUCell(2, 1)
        nac = NALU(2, 1, 2, 2, 10**-5)

        with tf.device("/gpu:0"):
            for i in range(50000):

                loss = loss_fn(nac, x, target_y)
                optim.minimize(lambda :loss_fn(nac, x, target_y))

                print("%.7f" % loss.numpy())

        # test case
        x = tf.constant([[1, 1], [2, 1], [3, 2], [11, 10]], dtype=tf.float32)
        y = nac(x)
        print(y.numpy())

























