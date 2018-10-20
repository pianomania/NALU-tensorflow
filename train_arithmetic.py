import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import os
from nalu import *
import argparse
import random


tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='train and test setting')
parser.add_argument('--train_size', default = 1000)
parser.add_argument('--test_size', default = 500)
parser.add_argument('--range_max', default = 50)
parser.add_argument('--range_min', default = 0)
parser.add_argument('--max_epochs', default = 50000)
parser.add_argument('--extened_range', default = 50)
parser.add_argument('--model', default = 'nalu')
parser.add_argument('--arithmetic', default = 'add')
parser.add_argument('--hidden_size', default = 2)
parser.add_argument('--eps', default = 1e-5)
parser.add_argument('--n_stack', default = 2)
parser.add_argument('--lr', default = 0.1)
parser.add_argument('--beta1', default = 0.9)
parser.add_argument('--beta2', default = 0.999)
parser.add_argument('--train', default = True)


arithmetics = {
    "add": lambda x: tf.reshape((x[:,0] + x[:,1]),[-1, 1]),
    "sub":  lambda x: tf.reshape((x[:,0] - x[:,1]),[-1, 1]),
    "mul": lambda x: tf.reshape((x[:,0] * x[:,1]),[-1, 1]),
    "div": lambda x: tf.reshape((x[:,0] / x[:,1]),[-1, 1]),
    "square": lambda x: tf.pow(x, 2),
    "sqrt": lambda x: tf.sqrt(x)
}


def loss_fn(model, x, target_y):
    y = model(x)
    return tf.reduce_mean(tf.pow(target_y - y, 2))


def train_arithmetic_task(args, model, dataset, arithmetic):
    

    max_epochs = args.max_epochs
    math_key, _ = arithmetic
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2

    x, target_y = dataset['train']

    #remove files in logs
    list_logs = os.listdir("./logs")
    for log in list_logs:
        os.remove("./logs/"+log)

    # tensorboard
    writer = tf.contrib.summary.create_file_writer("./logs")
    writer.set_as_default()

    global_step=tf.train.get_or_create_global_step()  # return global step var
    lr = tf.train.linear_cosine_decay(
        lr,
        global_step,
        max_epochs,
        num_periods=0.5,
        alpha=0.0,
        beta=0.001,
        name=None)
    optim = tf.train.AdamOptimizer(lr, beta1, beta2)

    # saver
    saver = tf.contrib.eager.Saver(model.var_list)

    for i in range(max_epochs):

        global_step.assign_add(1)

        with tf.contrib.summary.record_summaries_every_n_global_steps(10):
            loss = loss_fn(model, x, target_y)
            optim.minimize(lambda :loss_fn(model, x, target_y))

            print("{0}: {1:.7f}".format(i, loss.numpy()))

            tf.contrib.summary.scalar('loss', loss)
    
    saver.save('./model/{0}_{1}/model'.format(args.model, math_key))


def test_arithmetic_task(args, model, dataset, arithmetic):
    '''
    return:  mean square error of interpolation and extrapolation
    '''


    math_key, _ = arithmetic

    x_inter, target_y_inter = dataset['inter']
    x_extra, target_y_extra = dataset['extra']

    # saver and reload model
    saver = tf.contrib.eager.Saver(model.var_list)
    saver.restore(tf.train.latest_checkpoint(
        './model/{0}_{1}'.format(args.model, math_key)))

    # interpolation test
    predict_y_inter = model(x_inter)
    mse_inter = tf.reduce_mean(tf.square(predict_y_inter - target_y_inter))

    # extrapolation test
    predict_y_extra = model(x_extra)
    mse_extra = tf.reduce_mean(tf.square(predict_y_extra - target_y_extra))

    return mse_inter.numpy(), mse_extra.numpy()

def arithmetic_dataset(args, arithmetic):
    
    train_size = args.train_size
    test_size = args.test_size
    range_max = args.range_max
    range_min = args.range_min
    extened_range = args.extened_range

    math_key, math_op = arithmetic

    if math_key == "sqrt" or math_key == "square":
        in_dim = 1
    else:
        in_dim = 2

    # train dataset
    x = tf.random_uniform([train_size, in_dim], range_min, range_max)
    target_y = math_op(x)

    # interpolation test 
    x_inter = tf.random_uniform([test_size, in_dim], range_min, range_max)
    target_y_inter = math_op(x_inter)

    # extrapolation test
    if math_key == "sqrt" or math_key == "square":
         x_extra = tf.random_uniform([test_size, in_dim], range_max, range_max + extened_range)
    else:
        x_extra = np.random.uniform(range_min, range_max, [test_size, in_dim])
        idx = np.random.randint(0, in_dim, [test_size])
        x_extra[list(range(test_size)), idx] = np.random.uniform(
            range_max, range_max + extened_range, [test_size])
        x_extra = tf.constant(x_extra,  dtype=tf.float32)

    target_y_extra = math_op(x_extra)

    dataset = {
        'train': (x, target_y),
        'inter': (x_inter, target_y_inter),
        'extra': (x_extra, target_y_extra)
    }

    return dataset


if __name__ == "__main__":

    args = parser.parse_args()

    math_key = args.arithmetic
    math_op = arithmetics[math_key]
    hidden_size = args.hidden_size
    n_stack = args.n_stack
    eps = args.eps


    if math_key == "sqrt" or math_key == "square":
        in_dim = 1
    else:
        in_dim = 2 

    if args.model == 'nalu':
        model = NALU(in_dim, 1, hidden_size, n_stack, eps)
    elif args.model == 'nac':
        model = NAC(in_dim, 1, hidden_size, n_stack)

    dataset = arithmetic_dataset(args, (math_key, math_op))

    if args.train == True:
        train_arithmetic_task(args, model, dataset, (math_key, math_op))
    else:
        inter_mse, extra_mse = test_arithmetic_task(args, model, dataset, (math_key, math_op))
        print("Interpolation mse: {0}".format(inter_mse))
        print("extrapolation mse: {0}".format(extra_mse))
