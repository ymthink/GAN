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
    def __init__(self):
        self.layer_num = 0
        self.weights = []
        self.biases = []

    def _conv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('conv'+str(self.layer_num)):
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
    def _deconv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('deconv'+str(self.layer_num)):
            input_shape = tf.shape(input)

            init_w = he_init([filter_size, filter_size, output_dim, input_shape[3]], stride)
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
        with tf.variable_scope('dense'+str(self.layer_num)):
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

    def _residual_block(self, input, output_dim, filter_size, n_blocks=5):
        output = input
        with tf.variable_scope('residual_block'):
            for i in range(n_blocks):
                bypass = output
                output = self._deconv2d(output, output_dim, filter_size, 1)
                output = self._batch_norm(output)
                output = tf.nn.relu(output)

                output = self._deconv2d(output, output_dim, filter_size, 1)
                output = self._batch_norm(output)
                output = tf.add(output, bypass)

        return output

    def pixel_shuffle(x, r, n_split):
        def PS(x, r):
            bs, a, b, c = x.get_shape().as_list()
            x = tf.reshape(x, (bs, a, b, r, r))
            x = tf.transpose(x, (0, 1, 2, 4, 3))
            x = tf.split(x, a, 1)
            x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
            x = tf.split(x, b, 1)
            x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
            return tf.reshape(x, (bs, a*r, b*r, 1))

        xc = tf.split(x, n_split, 3)
        return tf.concat([PS(x_, r) for x_ in xc], 3)


class SRGAN(object):
    def __init__(self, batch_size, learning_rate, LAMBDA, SIGMA):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vgg = VGG19(None, None, None)
        self.LAMBDA = LAMBDA
        self.SIGMA = SIGMA

    def _generator(self, z):
        G = Network()
        #Network._deconv2d(input, output_dim, filter_size, stride)
        h = tf.nn.relu(G._deconv2d(z, 64, 3, 1))
        bypass = h

        h = G._residual_block(h, 64, 3, 5)

        h = G._deconv2d(h, 64, 3, 1)
        h = G._batch_norm(h)
        h = tf.add(h, bypass)

        h = G._deconv2d(h, 256, 3, 1)
        h = G._pixel_shuffle(h, 2, 64)
        h = tf.nn.relu(h)

        h = G._deconv2d(h, 64, 3, 1)
        h = G._pixel_shuffle(h, 2, 16)
        h = tf.nn.relu(h)

        h = G._deconv2d(h, 3, 3, 1)
        
        params = G.weights+G.biases

        return h, params

    def _discriminator(self, x):
        D = Network()
        #_conv2d(input, output_dim, filter_size, stride, padding='SAME')
        h = D._conv2d(x, 64, 3, 1)
        h = lrelu(h)

        h = D._conv2d(h, 64, 3, 2)
        h = lrelu(h)
        h = D._batch_norm(h)

        map_nums = [128, 256, 512]

        for map_num in map_nums:
            h = D._conv2d(h, map_num, 3, 1)
            h = lrelu(h)
            h = D._batch_norm(h)

            h = D._conv2d(h, map_num, 3, 2)
            h = lrelu(h)
            h = D._batch_norm(h)

        h = D._dense(h, 1024)
        h = lrelu(h)

        h = D._dense(h, 1)

        params = D.weights+D.biases
        
        return h, params
    def _downscale(self, x, K):
        mat = np.zeros([K, K, 3, 3])
        mat[:, :, 0, 0] = 1.0 / K**2
        mat[:, :, 1, 1] = 1.0 / K**2
        mat[:, :, 2, 2] = 1.0 / K**2
        filter = tf.constant(mat, dtype=tf.float32)
        return tf.nn.conv2d(x, filter, strides[1, K, K, 1], padding='SAME')

    def _creat_model(self):
        self.x = tf.placeholder(
            tf.float32,
            [None, self.height, self.width, 3],
            name='x'
        )
        self.z = self._downscale(self.x, 4)

        with tf.variable_scope('generator'):
            self.g = self._generator(self.z)
        with tf.variable_scope('discriminator') as scope:
            self.D_real = self._discriminator(self.x)
            scope.variable_reuse()
            self.D_fake = self._discriminator(self.g)

        _, real_phi = self.vgg.build_model(self.x, tf.constant(False), False)
        _, fake_phi = self.vgg.build_model(self.g, tf.constant(False), True)

        loss = None
        for i in range(len(real_phi)):
            l2_loss = tf.nn.l2_loss(real_phi[i] - fake_phi[i])
            if loss is None:
                loss = l2_loss
            else:
                loss += l2_loss

        content_loss = reduce_mean(loss)

        disc_loss = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        gen_loss = -tf.reduce_mean(self.D_fake)

        alpha = tf.random_uniform(
            shape=[self.batch_size,1],
            minval=0.,
            maxval=1.
        )

        differences = self.g - self.x
        interpolates = self.x + alpha * differences
        gradients = tf.gradients(self._discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.D_loss = self.SIGMA * (disc_loss + self.LAMBDA * gradient_penalty)

        self.G_loss = content_loss + self.SIGMA * gen_loss


        





        




