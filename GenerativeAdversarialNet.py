# -*- coding:utf-8 -*-
"""
A generative adverarial net class example.
Use MNIST set as samples.

Author:ymthink
E-mail:yinmiaothink@gmail.com
Date:May,8,2017
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GenerativeAdversarialNet(object):
    
    def __init__(self, gen_shape, dis_shape, batch_size, 
        step_num, learning_rate, data_width, data_length, data):

        self.gen_shape = gen_shape
        self.dis_shape = dis_shape
        self.batch_size = batch_size
        self.step_num = step_num
        self.learning_rate = learning_rate
        self.data_width = data_width
        self.data_length = data_length
        self.data = data
        self.gen_W = []
        self.gen_b = []
        self.dis_W = []
        self.dis_b = []

        self._creat_vars()
        self._creat_model()

    def _optimizer(self, loss, var_list):
        decay = 0.96
        decay_step_num = self.batch_size // 5
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            batch,
            decay_step_num,
            decay,
            staircase=True
        )
        opt = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=batch, var_list=var_list)
        return opt

    def _generator(self, z):
        current_input = z
        for i in range(len(self.gen_W)):
            current_output = tf.nn.sigmoid(tf.matmul(current_input, self.gen_W[i]) + self.gen_b[i])
            current_input = current_output
        return current_output

    def _discriminator(self, x):
        current_input = x
        for i in range(len(self.dis_W)):
            current_output = tf.nn.sigmoid(tf.matmul(current_input, self.dis_W[i]) + self.dis_b[i])
            current_input = current_output
        return current_output

    def _creat_vars(self):
        gen_len = len(gen_shape)
        dis_len = len(dis_shape)
        #Generator
        with tf.variable_scope('Generator'):
            for gen_i in range(gen_len-1):
                W = tf.Variable(
                    tf.random_normal([gen_shape[gen_i], gen_shape[gen_i+1]]), 
                    name='W'+str(gen_i)
                )
                b = tf.Variable(tf.zeros([gen_shape[gen_i+1]]), name='b'+str(gen_i))
                self.gen_W.append(W)
                self.gen_b.append(b)

        #Discriminator
        with tf.variable_scope('Discriminator'):
            for dis_i in range(dis_len-1):
                W = tf.Variable(
                    tf.random_normal([dis_shape[dis_i], dis_shape[dis_i+1]]), 
                    name='W'+str(dis_i)
                )
                b = tf.Variable(tf.zeros([dis_shape[dis_i+1]]), name='b'+str(dis_i))
                self.dis_W.append(W)
                self.dis_b.append(b)

    def _creat_model()
        self.z = tf.placeholder([None, gen_shape[0]], name='z')
        self.x = tf.placeholder([None, dis_shape[0], name='x')

        self.g = self._generator(self.z)
        self.D_x = self._discriminator(self.x)
        self.D_g = self._discriminator(self.g)

    def _display(self, display_num):
    zs = np.random.uniform(-1., 1., size=[display_num, self.gen_shape[0]])
    gs = self.sess.run(self.g, feed_dict={self.z:zs})

        fig, ax = plt.subplots(2, display_num)
        for fig_i in range(display_num):
            ax[0][fig_i].imshow(np.reshape(data.test.images[fig_i], (self.data_length, self.data_width)))
            ax[0][fig_i].set_xticks([])
            ax[0][fig_i].set_yticks([])

            ax[1][fig_i].imshow(np.reshape(gs[fig_i], (self.data_length, self.data_width)))
            ax[1][fig_i].set_xticks([])
            ax[1][fig_i].set_yticks([])
        plt.show()

    def train(self):
        loss_dis = -tf.reduce_mean(tf.log(self.D_x) + tf.log(1 - self.D_g))
        loss_gen = -tf.reduce_mean(tf.log(self.D_g))
        opt_dis = self._optimizer(loss_dis, var_list=dis_W+dis_b)
        opt_gen = self._optimizer(loss_gen, var_list=gen_W+gen_b)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        sess.run(init)
        
        disp_step_num = int(self.step_num / 20)
        dispay_num = 10

        for step in range(self.step_num):
            xs, ys = self.data.train.next_batch(batch_size)
            zs = np.random.uniform(-1., 1., size=[batch_size, self.gen_shape[0]])
            _, l_dis = self.sess.run([opt_dis, loss_dis], feed_dict={self.z:zs, self.x:xs})
            _, l_gen = self.sess.run([opt_gen, loss_gen], feed_dict={self.x:xs})

            if step % disp_step_num == 0:
            print('Step:', '%d'%(step+1), 'loss_dis =', '{:.9f}'.format(l_dis), 'loss_gen = ', '{:.9f}'.format(l_gen))

        self._display(display_num)
        self.sess.close()

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("MNIST_data", one_hot=True)

    learning_rate = 0.001
    step_num = 20000
    batch_size = 256
    gen_shape = [100, 256, 784]
    dis_shape = [784, 256, 1]

    ae = GenerativeAdversarialNet(
        gen_shape=gen_shape,
        dis_shape=dis_shape,
        batch_size=batch_size,
        step_num=step_num,
        learning_rate=learning_rate,
        data_width=28,
        data_length=28,
        data=data
    )















