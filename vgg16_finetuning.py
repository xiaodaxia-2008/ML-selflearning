########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import os
import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
#from imagenet_classes import class_names
from tensorflow.examples.tutorials.mnist import input_data

class_names = ['cat', 'dog']

class vgg16:
    def __init__(self, imgs, labels, weights=None, sess=None):
        self.imgs = imgs
        self.labels = labels
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc4l)
        self.training()
        self.acc()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]
        
        with tf.name_scope('fc4') as scope:
            fc4w = tf.Variable(tf.truncated_normal([1000, 2], dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc4b = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc4l = tf.nn.bias_add(tf.matmul(self.fc3l, fc4w), fc4b)
            

    def load_weights(self, weight_file, sess):
        sess.run(tf.global_variables_initializer())
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def training(self):
        with tf.name_scope('train') as scope:
#            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
#                    logits=self.fc4l, labels=self.labels))
            self.loss = tf.losses.softmax_cross_entropy(self.labels, self.fc4l)
            optimizer = tf.train.AdamOptimizer(0.001)
            fc4parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              scope='fc4')
            self.train_op = optimizer.minimize(self.loss, var_list=fc4parameters)

    def run_training(self, max_steps, sess, datasets, batch_size):
#        sess.run(tf.global_variables_initializer()) # load_weights already do this
        saver = tf.train.Saver()
        for step in range(max_steps):
            starttime = time.time()
            xs, ys = next(datasets.train_data())
            feed_dict = {self.imgs: xs, self.labels: ys}
            _, loss_ = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            duration = time.time() - starttime
            if step % 10 == 0 or (step + 1) == max_steps:
                check_point_file = os.path.join('./model', 'model.ckpt')
                saver.save(sess, check_point_file, global_step=step)
                xs, ys = datasets.validation_data()
                feed_dict = {self.imgs: xs, self.labels: ys}
                loss_, accura = sess.run([self.loss, self.acc], 
                                         feed_dict=feed_dict)
                print(f"step:{step}, duration:{duration}, loss:{loss_}, ",
                      f"acc:{accura/datasets.valid_batch_size}")

    def acc(self):
        with tf.name_scope('accuracy') as scope:
            correct_predict = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.labels, 1))
            self.acc = tf.reduce_sum(tf.cast(correct_predict, 'int32'))

    def predict(self, imgpath, sess):
        img = imread(imgpath, mode='RGB')
        img = imresize(img, (224, 224))
        probs = sess.run(self.probs, feed_dict={self.imgs: [img]})[0]
        return class_names[np.argmax(probs)]


class Data:
    def __init__(self, dir_path, batch_size=None):
        self.dir_path = dir_path
        self.img_width = 224
        self.img_height = 224
        self.channels = 3
        self.num_classes = 2
        self.file_list = os.listdir(self.dir_path)
        np.random.shuffle(self.file_list)
        self.files_num = len(self.file_list)
        self.valid_batch_size = 50
        if batch_size is None:
            self.batch_size = files_num
        else:
            self.batch_size = batch_size
            
    def train_data(self):   
        xdata = np.zeros((self.batch_size, self.img_width, 
                          self.img_height, self.channels), dtype=np.float32)
        ydata = np.zeros((self.batch_size, self.num_classes), dtype=np.int32)
        count = 0
        file_i = 0
        while True:
            file = self.file_list[file_i % self.files_num]
            img = imread(os.path.join(self.dir_path, file), mode='RGB')
            img = imresize(img, (224, 224))
            label = np.array([1, 0]) if ('cat' in file) else np.array([0, 1])
            xdata[count, :] = img
            ydata[count, :] = label
            count += 1
            file_i += 1
            if count == self.batch_size:
                yield xdata, ydata
                count = 0
    
    def validation_data(self):
        xdata = np.zeros((self.valid_batch_size, self.img_width, 
                          self.img_height, self.channels), dtype=np.float32)
        ydata = np.zeros((self.valid_batch_size, self.num_classes), dtype=np.int32)
        choices = np.random.choice(self.file_list, size=self.valid_batch_size)
        count = 0
        for file in choices:
            img = imread(os.path.join(self.dir_path, file), mode='RGB')
            img = imresize(img, (224, 224))
            label = np.array([1, 0]) if ('cat' in file) else np.array([0, 1])
            xdata[count, :] = img
            ydata[count, :] = label
            count += 1
        return xdata, ydata

if __name__ == '__main__':
#    img1 = imread('data/train/cat.1.jpg', mode='RGB')
#    img1 = np.reshape(imresize(img1, (224, 224)), (-1, 224, 224, 3))
#    label = np.zeros((1, 2), dtype=np.int32)
#    label[0, 0] = 1
    check_point_file = os.path.join('./model', 'model.ckpt')
    BATCH_SIZE = 20
    datasets = Data('data/train', batch_size=BATCH_SIZE)
    
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [None, 2])
    vgg = vgg16(imgs, labels, 'vgg16_weights.npz', sess)
    saver = tf.train.Saver()
    ckpt_dir = './model'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, './model/model.ckpt-99')
    # summary_writer = tf.summary.FileWriter('./log', graph=sess.graph)
    # vgg.run_training(100, sess, datasets=datasets, batch_size=BATCH_SIZE)


#    images, img_labels = next(datasets)
#    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: images})
#    preds = np.argmax(prob, axis=1)
#    true_labels = np.argmax(img_labels, axis=1)
#    for p, t in zip(preds, true_labels):
#        print(class_names[p], class_names[t])
#    sess.close()
    # summary_writer.close()
