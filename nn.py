import numpy as np  # linear algebra
import tensorflow as tf
import os
import random
import nibabel as nib
from keras.utils import to_categorical
from sys import getsizeof

n_classes = 2  # from 0 to 9, 10 labels totally
#width = 114
width = 38
#height = 155
height = 52 
# depth = 103
depth = 34

with tf.name_scope('inputs'):
    x_input = tf.placeholder(tf.float32, shape=[None, width, height, depth, 1])
    y_input = tf.placeholder(tf.float32, shape=[None, n_classes])


def get_data(data_set, amount_of_data_needed):
    """We assume that data_set folder is located in current folder."""
    global n_classes
    global sess
    filenames = []
    for s in os.listdir(data_set):
        if s[:3] == 'LGG':
            filenames.append([s, 1])  # 1 is LGG
        elif s[:3] == 'HGG':
            filenames.append([s, 0])  # 0 is HGG
    x_data = []
    y_data = []
    random.shuffle(filenames)
    for x, y in filenames[:amount_of_data_needed]:
        img = nib.load(data_set + '/' + x)
        pic_data = img.get_fdata()
        img.uncache()
        pic_data = tf.image.resize_images(pic_data, size=(width, height), method=1)
        pic_data = tf.Session().run(pic_data)
        
        pic_data = tf.image.resize_images(np.rot90(pic_data, 1, (0, 2)), size=(depth, height), method=1)
        pic_data = tf.Session().run(pic_data)
        pic_data = pic_data.reshape(width, height, depth, 1)
        
        print(pic_data.shape)
        print()
        x_data.append(pic_data)
        y_data.append(y)
    
    return np.asarray(x_data, dtype=np.float32), to_categorical(y_data, n_classes)

def cnn_model(x_train_data, keep_rate=0.7, seed=None):

    with tf.name_scope("layer_a"):
        # conv => 16*16*16
        data = tf.layers.conv3d(inputs=x_train_data, filters=2, kernel_size=[3, 3, 3], strides = 1, padding='same', activation=tf.nn.relu)
        print("conv1: {}",format(getsizeof(data) / 1024 / 1024))
        # # conv => 16*16*16
        data = tf.layers.conv3d(inputs=data, filters=4, kernel_size=[3, 3, 3], strides = 1, padding='same', activation=tf.nn.relu)
        # # pool => 8*8*8
        print("conv2: {}",format(getsizeof(data) / 1024 / 1024))
        data = tf.layers.max_pooling3d(
            inputs=data, pool_size=[2, 2, 2], strides=[2, 2, 2])
        print("pool1: {}",format(getsizeof(data) / 1024 / 1024))

    with tf.name_scope("layer_c"):
        # conv => 8*8*8
        data = tf.layers.conv3d(inputs=data, filters=8, kernel_size=[3, 3, 3], strides = 1, padding='same', activation=tf.nn.relu)
        print("conv3: {}",format(getsizeof(data) / 1024 / 1024))
        # # conv => 8*8*8
        # conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[
        #                          3, 3, 3], padding='same', activation=tf.nn.relu)
        # #pool => 4*4*4
        data = tf.layers.max_pooling3d(inputs=data, pool_size=[2, 2, 2], strides=[2, 2, 2])
        print("pool2: {}",format(getsizeof(data) / 1024 / 1024))
    
    with tf.name_scope("batch_norm"):
        data = tf.layers.batch_normalization(inputs=data, training=True)
        print("bn: {}",format(getsizeof(data) / 1024 / 1024))
    with tf.name_scope("fully_con"):
        data = tf.reshape(data, [-1, 8 * 8 * width // 4 * height // 4 * depth // 4])
        print("flattering: {}",format(getsizeof(data) / 1024 / 1024))
        data = tf.layers.dense(inputs=data, units=1024, activation=tf.nn.relu)
        print("dense: {}",format(getsizeof(data) / 1024 / 1024))
        # (1-keep_rate) is the probability that the node will be kept
        data = tf.layers.dropout(inputs=data, rate=keep_rate, training=True)
        print("drouput: {}",format(getsizeof(data) / 1024 / 1024))

    with tf.name_scope("y_conv"):
        y_conv = tf.layers.dense(inputs=data, units=2)
        print("dense: {}",format(getsizeof(y_conv) / 1024 / 1024))

    return y_conv


def train_neural_network(x_train_data, y_train_data, x_test_data, y_test_data, learning_rate=0.05, keep_rate=0.7, epochs=10, batch_size=32):

    with tf.name_scope("cross_entropy"):
        prediction = cnn_model(x_input, keep_rate, seed=1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction, labels=y_input))

    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    iterations = int(len(x_train_data) / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import datetime

        start_time = datetime.datetime.now()

        iterations = int(len(x_train_data) / batch_size)
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch', epoch, 'started', end='')
            epoch_loss = 0
            # mini batch
            for itr in range(iterations):
                mini_batch_x = x_train_data[itr * batch_size: (itr + 1) * batch_size]
                mini_batch_y = y_train_data[itr * batch_size: (itr + 1) * batch_size]
                _optimizer, _cost = sess.run([optimizer, cost], feed_dict={ x_input: mini_batch_x, y_input: mini_batch_y})
                epoch_loss += _cost

            #  using mini batch in case not enough memory
            acc = 0
            itrs = int(len(x_test_data) / batch_size)
            for itr in range(itrs):
                mini_batch_x_test = x_test_data[itr * batch_size: (itr + 1) * batch_size]
                mini_batch_y_test = y_test_data[itr * batch_size: (itr + 1) * batch_size]
                acc += sess.run(accuracy, feed_dict={ x_input: mini_batch_x_test, y_input: mini_batch_y_test})

            end_time_epoch = datetime.datetime.now()
            print(' Testing Set Accuracy:', acc / itrs, ' Time elapse: ',
                  str(end_time_epoch - start_time_epoch))

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

# Get data from dataset
#dir = 'drive/temp/'
dir = ''
train_dir = dir + 'train'
test_dir = dir + 'test'
(x_train, y_train) = get_data(train_dir, 10)
(x_test, y_test) = get_data(test_dir, 10)


train_neural_network(x_train, y_train, x_test, y_test, epochs=10, batch_size=1, learning_rate=0.3)

