import tensorflow as tf


var_init = lambda x: tf.variance_scaling_initializer(scale=x, distribution='truncated_normal')
scale = 0.1

class NACCell(object):

    def __init__(self, in_dim, out_dim):

        with tf.variable_scope('nac-cell') as vs:
            self.w = tf.get_variable(name = 'w', shape = [in_dim, out_dim], initializer=var_init(scale))
            self.m = tf.get_variable(name = 'm', shape = [in_dim, out_dim], initializer=var_init(scale))

        self.var_list = [self.w, self.m]

    def __call__(self, x):

        w = tf.multiply(tf.tanh(self.w), tf.sigmoid(self.m))
        y = tf.matmul(x, w)

        return y


class NAC(object):

    def __init__(self, in_dim=2, out_dim=1, hidden_dim=2, n_stacks=2):

        self.layers = []
        self.var_list = []

        for i in range(n_stacks):
            with tf.variable_scope('nac_' + str(i)) as vs:
                self.layers.append(
                    NACCell(
                        in_dim if i == 0 else hidden_dim, 
                        out_dim if i == n_stacks-1 else hidden_dim
                    )
                )

        for layer in self.layers:
            self.var_list += layer.var_list

    def __call__(self, x):

        y = x

        for layer in self.layers:
            y = layer(y)

        return y


class NALUCell(object):

    def __init__(self, in_dim, out_dim, eps=1e-5):
        
        with tf.variable_scope("nalu-cell-add") as vs:
            self.add_cell = NACCell(in_dim, out_dim)
        with tf.variable_scope("nalu-cell-mul") as vs:
            self.mul_cell = NACCell(in_dim, out_dim)
        with tf.variable_scope("nalu-cell-g") as vs:
            self.G = tf.get_variable('g', shape = [in_dim, out_dim], initializer=var_init(scale))
        self.eps = eps
        
        self.var_list = self.add_cell.var_list + self.mul_cell.var_list + [self.G]

    def __call__(self, x):

        a = self.add_cell(x)
        m = self.mul_cell(tf.log(tf.abs(x) + self.eps))
        m = tf.exp(m)
        g = tf.sigmoid(tf.matmul(x, self.G))
        y = tf.multiply(g, a) + tf.multiply(1-g, m)

        return y

class NALU(object):

    def __init__(self, in_dim, out_dim, hidden_dim, n_stacks, eps):

        self.layers = []
        self.var_list = []

        for i in range(n_stacks):
            with tf.variable_scope('nalu_' + str(i)) as vs:
                self.layers.append(
                    NALUCell(
                        in_dim if i == 0 else hidden_dim,
                        out_dim if i == n_stacks-1 else hidden_dim
                    )
                )

        for layer in self.layers:
            self.var_list += layer.var_list

    def __call__(self, x):

        y = x

        for layer in self.layers:
            y = layer(y)

        return y
