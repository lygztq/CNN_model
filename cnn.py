# internal imports
import tensorflow as tf
import os
"""CNN definition."""


def get_default_hparams():
    """Return default HParams for cnn."""
    hparams = tf.contrib.training.HParams(
        data_set_path='dataSet',  # Our dataset.
        train_steps=110,  # Total number of steps of training. Keep large.
        output_size=2,   # The dimensionality of the output
        input_img_size=80,  # The size of input image
        batch_size=30,  # Train batch size
        initial_learning_rate=0.001,  # The initial learning rate
        learning_rate_decay_rate=0.7,  # The decay of learning rate
        w_weight=0.1,  # The weight of the W cost
        # 'c':convolution, 'p':pooling, 'f':fully connection layer, 'd':drop, 'r':reLU
        # ['c',index,shape], ['p',index,mode], ['f',index,shape,reshape], ['d',index,keep_prob], ['r',index]
        layers=[
            ['c', 1, [5, 1, 32]],
            ['p', 1, 'max'],
            ['c', 2, [5, 32, 64]],
            ['p', 2, 'max'],
            ['c', 3, [5, 64, 64]],
            ['p', 3, 'max'],
            ['f', 1, [10*10*64, 1024], True],
            ['r', 1],
            ['d', 1, 0.5],
            ['f', 2, [1024, 2], False]
        ],  # layers and params
        # keep_prob=0.5,  # the probability of keeping unit
        save_path='models',
        # min_learning_rate=0.0001,  # Minimum learning rate.
    )
    return hparams


def weight_variable(shape):
    """generate the initial value of weight of convolution core"""
    # truncated_normal(shape,...) truncated normal distribution
    # used for generate a small positive initial value
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """generate the initial value of bias of convolution core"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """convolution operation"""
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max pooling operation"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def ave_pool_2x2(x):
    """average pooling operation"""
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def save_model(sess, path, step):
    """save graph and model data"""
    saver = tf.train.Saver(tf.global_variables())
    save_path = os.path.join(path, 'vector')
    tf.logging.info('saving model %s.', save_path)
    tf.logging.info('global_step %i.', step)
    saver.save(sess, save_path, global_step=step)


class ConvLayer(object):
    """convolution layer class, containing ReLu"""
    def __init__(self, size, input_dim, output_dim, input_image):
        self._shape = [size, size, input_dim, output_dim]
        self._conv_core = weight_variable(self._shape)
        self._bias = bias_variable([output_dim])
        self._output_image = tf.nn.relu(conv2d(input_image, self._conv_core)+self._bias)

    @property
    def shape(self):
        return self._shape

    @property
    def core(self):
        return self._conv_core

    @property
    def bias(self):
        return self._bias

    @property
    def output(self):
        return self._output_image


class PoolingLayer(object):
    """pooling layer class"""
    def __init__(self, input_image, mode='max'):
        if mode == 'max':
            self._output = max_pool_2x2(input_image)
        elif mode == 'avg':
            self._output = ave_pool_2x2(input_image)

    @property
    def output(self):
        return self._output


class DCLayer(object):
    """fully connected layer class"""
    def __init__(self, input_vector, input_size, output_size, reshape=True):
        self._fcWeight = weight_variable([input_size, output_size])
        self._fcBias = bias_variable([output_size])
        if reshape:
            input_vector = tf.reshape(input_vector, [-1, input_size])
        self._output = tf.matmul(input_vector, self._fcWeight) + self._fcBias

    @property
    def output(self):
        return self._output


class ReLuLayer(object):
    """ReLu Layer class"""
    def __init__(self, input_img):
        self._output = tf.nn.relu(input_img)

    @property
    def output(self):
        return self._output


class DropoutLayer(object):
    """Dropout layer class"""
    def __init__(self, input_img, keep_prob):
        self._output = tf.nn.dropout(input_img, keep_prob)

    @property
    def output(self):
        return self._output


class Model(object):
    """define a CNN model"""

    def __init__(self, hps, reuse=False):
        self.hps = hps
        with tf.variable_scope('cnn', reuse=reuse):
            tf.logging.info('model initialization')
            # Define input and some weights.
            self.layer_set = []
            self.input_data = tf.placeholder(dtype=tf.float32,
                                             shape=[None, hps.input_img_size * hps.input_img_size],
                                             name="input")
            self.labels = tf.placeholder(dtype=tf.float32,
                                         shape=[None, hps.output_size],
                                         name="label")
            self.input_image = tf.reshape(self.input_data,
                                          shape=[-1, hps.input_img_size, hps.input_img_size, 1])
            self.W_total_L2 = tf.zeros(shape=[1], dtype=tf.float32)
            self.build_model(hps)

    def build_model(self, hps):
        # structure of CNN
        tf.logging.info('Building model')
        for layer in hps.layers:
            # Which layer type.
            layer_type = layer[0]
            layer_index = layer[1]
            if layer_type == 'c':
                layer_shape = layer[2]
                if layer_index == 1:
                    new_layer = ConvLayer(layer_shape[0], layer_shape[1], layer_shape[2], self.input_image)
                else:
                    new_layer = ConvLayer(layer_shape[0], layer_shape[1], layer_shape[2], self.layer_set[-1].output)
                self.W_total_L2 = self.W_total_L2 + tf.norm(new_layer.core)
            elif layer_type == 'p':
                mode = layer[2]
                new_layer = PoolingLayer(self.layer_set[-1].output, mode)
            elif layer_type == 'f':  # ['f',index,shape,reshape]
                shape = layer[2]
                reshape = layer[3]
                new_layer = DCLayer(self.layer_set[-1].output, shape[0], shape[1], reshape)
            elif layer_type == 'r':
                new_layer = ReLuLayer(self.layer_set[-1].output)
            elif layer_type == 'd':  # ['d',index,keep_prob]
                prob = layer[2]
                new_layer = DropoutLayer(self.layer_set[-1].output,prob)
            self.layer_set.append(new_layer)

        # output
        self.train_result = self.layer_set[-1].output
        self.result = tf.argmax(self.layer_set[-1].output, 1, name="result")

        # Loss function
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.train_result))
        self.Loss = hps.w_weight*self.W_total_L2 + self.cross_entropy

        # train
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.learning_rate = tf.train.exponential_decay(hps.initial_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_rate=hps.learning_rate_decay_rate,
                                                        decay_steps=hps.train_steps)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Loss)
        self.correct_prediction = tf.equal(tf.argmax(self.train_result, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="acc")

