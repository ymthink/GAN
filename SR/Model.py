'''An example for image SR using GAN
<Photo-Realistic Single Image Super-Resolution 
Using a Generative Adversarial Network>

https://arxiv.org/abs/1609.04802

E-mail: yinmiaothink@gmail.com
'''
import numpy as np
import tensorflow as tf

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def relu(x):
    return tf.nn.relu(x)

def elu(x):
    return tf.nn.elu(x)

def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)

def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size**2
    fan_out = output_dim * filter_size**2 / (stride**2)
    stddev = tf.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)

class Network(object):
    def __init__(self, name):
        self.name = name
        self.layer_num = 0
        self.weights = []
        self.biases = []

    def _conv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope(str(self.layer_num)):
            input_shape = tf.shape(input)

            init_w = he_init([filter_size, filter_size, input_shape[3], output_dim], stride)
            weight = tf.get_variable(
                'weight', 
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d(
                input,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output
    def _conv2d_transpose(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope(str(self.layer_num)):
            input_shape = tf.shape(input)

            init_w = he_init([filter_size, filter_size, input_shape[3], output_dim], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d_transpose(
                value=input,
                filter=weight,
                output_shape=tf.stack([
                    input_shape[0], 
                    input_shape[1]*stride, 
                    input_shape[2]*stride, 
                    output_dim
                ]),
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def _upscale(self, input):
        input_shape = tf.shape(input)
        output = tf.image_resize_nearest_neighbor(input, [2 * s for s in input_shape[1:3]])

        return output

    def _batch_norm(self, input, scale=False):
        ''' batch normalization
        ArXiv 1502.03167v3 '''

        output = tf.contrib.layers.batch_norm(input, scale=scale)
        return output

    def _dense(self, input, output_dim):
        with tf.variable_scope(str(self.layer_num)):
            input_shape = tf.shape(input)

            init_w = xavier_init([input_shape[1], output_dim])
            weight = tf.get_variable('weight', initializer=init_w)

            init_b = tf.zeros([output_dim])
            bias = get_variable('bias', initializer=init_b)

            output = tf.add(tf.matmul(input, weight), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def _residual_block(self, input, output_dim, filter_size, n_layers=2):
        if output_dim != int(input.get_shape()[3]):
            output = self._conv2d(
                input=input, 
                output_dim=output_dim,
                filter_size=1,
                stride=1
            )
        else:
            output=input

        bypass = output

        for _ in range(n_layers):
            output = relu(self._batch_norm(output))
            output = self._conv2d(
                input=output, 
                output_dim=output_dim, 
                filter_size=filter_size,
                stride=1
            )

        return tf.add(bypass, output)


class SRGAN(object):
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.G = Network('Generator')
        self.D = Network('Discriminator')

    def _generator(self, z):
        output = z
        map_nums = [256, 128]
        for map_num in map_nums:
            for _ in range(2):
                output = self.G._residual_block(
                    input=output,
                    output_dim=map_num,
                    filter_size=3
                )

            output = relu(self.G._upscale(output))
            output = self.G._conv2d_transpose(
                input=output,
                output_dim=map_num,
                filter_size=3,
                stride=1
            )




        




