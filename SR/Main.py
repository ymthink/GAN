#encoding=utf-8

import sys
sys.path.append('./')

from Model import *

learning_rate = 1e-3
batch_size = 32
LAMBDA = 10
SIGMA = 1e-3

step_num = 10000

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("../MNIST_data", one_hot=True)

def train():
    gan = SRGAN(28, 28, 1, LAMBDA, SIGMA)

    D_opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gan.D_loss, var_list=gan.D_params)
    G_opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(gan.G_loss, var_list=gan.G_params)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    if tf.train.get_checkpoint_state('./backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest')

    for step in range(step_num):
        for _ in range(5):
            xs, _ = data.train.next_batch(batch_size)
            xs = np.reshape(xs, [-1, 28, 28, 1])
            _, l_d = sess.run([D_opt, gan.D_loss], feed_dict={gan.x:xs})

        xs, _ = data.train.next_batch(batch_size)
        xs = np.reshape(xs, [-1, 28, 28, 1])
        _, l_g = sess.run([G_opt, gan.G_loss], feed_dict={gan.x:xs})

        if step % 100 == 0:
            print('step: {}, D_loss = {:.5f}, G_loss = {:.5f}'.format(step, l_d, l_g))
            saver = tf.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)


if __name__ == '__main__':
    train()



    

