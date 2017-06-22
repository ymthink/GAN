# -*- coding:utf-8 -*-
"""
A wasserstain generative adverarial net class example.
Use MNIST set as samples.

Author:ymthink
E-mail:yinmiaothink@gmail.com
Date:May,12,2017
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

class WGAN_GP(object):
    
    def __init__(self, gen_shape, dis_shape, batch_size, 
        step_num, learning_rate, data_width, data_length, LAMBDA, data):

        self.gen_shape = gen_shape
        self.dis_shape = dis_shape
        self.batch_size = batch_size
        self.step_num = step_num
        self.learning_rate = learning_rate
        self.data_width = data_width
        self.data_length = data_length
        self.LAMBDA = LAMBDA
        self.data = data
        self.gen_W = []
        self.gen_b = []
        self.dis_W = []
        self.dis_b = []

        self._creat_model()

    def _generator(self, z):
        current_input = z
        for i in range(len(self.gen_W)-1):
            current_output = tf.nn.relu(tf.matmul(current_input, self.gen_W[i]) + self.gen_b[i])
            current_input = current_output
        current_output = tf.nn.sigmoid(tf.matmul(current_input, self.gen_W[len(self.gen_W)-1]) + self.gen_b[len(self.gen_W)-1])
        return current_output

    def _discriminator(self, x):
        current_input = x
        for i in range(len(self.dis_W)-1):
            current_output = tf.nn.relu(tf.matmul(current_input, self.dis_W[i]) + self.dis_b[i])
            current_input = current_output
        current_output = tf.matmul(current_input, self.dis_W[len(self.dis_W)-1]) + self.dis_b[len(self.dis_b)-1]
        return current_output
    
    def _xavier_init(self, size):
        in_dim = size[0]
        stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=stddev)

    def _creat_vars(self):
        gen_len = len(gen_shape)
        dis_len = len(dis_shape)
        #Generator
        with tf.variable_scope('Generator'):
            for gen_i in range(gen_len-1):
                W = tf.Variable(
                    self._xavier_init([self.gen_shape[gen_i],self.gen_shape[gen_i+1]]),
                    name='W'+str(gen_i)
                )
                b = tf.Variable(tf.zeros([gen_shape[gen_i+1]]), name='b'+str(gen_i))
                self.gen_W.append(W)
                self.gen_b.append(b)

        #Discriminator
        with tf.variable_scope('Discriminator'):
            for dis_i in range(dis_len-1):
                W = tf.Variable(
                    self._xavier_init([self.dis_shape[dis_i], self.dis_shape[dis_i+1]]),
                    name='W'+str(dis_i)
                )
                b = tf.Variable(tf.zeros([dis_shape[dis_i+1]]), name='b'+str(dis_i))
                self.dis_W.append(W)
                self.dis_b.append(b)

    def _creat_model(self):
        self.z = tf.placeholder(tf.float32, [None, gen_shape[0]], name='z')
        self.x = tf.placeholder(tf.float32, [None, dis_shape[0]], name='x')
        self._creat_vars()

        self.g = self._generator(self.z)
        self.D_real = self._discriminator(self.x)
        self.D_fake = self._discriminator(self.g)

    def _display(self):
        zs = self.sample_z(16, self.gen_shape[0])
        gs = self.sess.run(self.g, feed_dict={self.z:zs})

        fig = plt.figure(figsize=(4,4))
        graph = gridspec.GridSpec(4, 4)
        graph.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(gs):
            ax = plt.subplot(graph[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        return fig
    def sample_z(self, m, n):
        return np.random.uniform(-1., 1., size=[m,n])

    def train(self):
        loss_dis = -tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
        loss_gen = -tf.reduce_mean(self.D_fake)

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

        loss_dis += self.LAMBDA * gradient_penalty

        params_dis = self.dis_W + self.dis_b
        params_gen = self.gen_W + self.gen_b

        opt_dis = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(loss_dis, var_list=params_dis)
        opt_gen = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(loss_gen, var_list=params_gen)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        
        disp_step_num = 1000
        display_num = 10

        if not os.path.exists('out/'):
            os.makedirs('out/')
        fig_i = 0

        for step in range(self.step_num):
            for _ in range(5):
                xs, ys = self.data.train.next_batch(batch_size)
                zs = self.sample_z(self.batch_size, self.gen_shape[0])
                _, l_dis = self.sess.run([opt_dis, loss_dis], feed_dict={self.z:zs, self.x:xs})

            zs = self.sample_z(self.batch_size, self.gen_shape[0])
            _, l_gen = self.sess.run([opt_gen, loss_gen], feed_dict={self.z:zs})

            if step % 100 == 0:
                print(
                        'Step: {}, loss_dis = {:.5}, loss_gen = {:.5}'
                        .format(step, l_dis, l_gen)
                )
            if step % disp_step_num == 0:
                fig = self._display()
                plt.savefig('out/{}.png'.format(str(fig_i).zfill(3)), bbox_inches='tight')
                fig_i += 1
                plt.close(fig)

        self.sess.close()

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("MNIST_data", one_hot=True)

    learning_rate = 1e-4
    LAMBDA = 10
    step_num = 100000
    batch_size = 32
    gen_shape = [10, 128, 784]
    dis_shape = [784, 128, 1]

    ae = WGAN_GP(
        gen_shape=gen_shape,
        dis_shape=dis_shape,
        batch_size=batch_size,
        step_num=step_num,
        learning_rate=learning_rate,
        data_width=28,
        data_length=28,
        LAMBDA=LAMBDA,
        data=data
    )
    ae.train()

















