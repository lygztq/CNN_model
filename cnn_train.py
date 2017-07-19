import tensorflow as tf
import cnn
import os
import utils
import numpy as np

project_path = '/home/ztq/PycharmProjects/CNN_model'
dataSetPath = os.path.join(project_path,'dataSet')
savePath = os.path.join(project_path,'models')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataSetPath', dataSetPath, 'the path of dataSet(bin file).'
)
tf.app.flags.DEFINE_string(
    'savePath', savePath, 'saving model path.'
)



def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def train(sess, model, train_set, test_set, test_true_num, test_false_num):
    tf.logging.info('training')
    hps = model.hps

    for i in range(hps.train_steps):
        step = sess.run(model.global_step)
        _, curr_learning_rate = sess.run([model.add_global, model.learning_rate])
        batch = train_set.next_batch(hps.batch_size)
        feed = {
            model.input_data: batch[0],
            model.labels: batch[1]
        }
        feed_test = {
            model.input_data: test_set.data,
            model.labels: test_set.label
        }

        if i % 50 == 0:
            train_accuracy = model.accuracy.eval(feed)
            cost = model.Loss.eval(feed)
            W_cost = model.W_total_L2.eval(feed)
            cross_cost = model.cross_entropy.eval(feed)
            print('step %d, training accuracy %g, cost: %d' % (i, train_accuracy, cost))
            print('W_cost: %g, cross_cost: %g, learning rate: %g\n' % (W_cost, cross_cost, curr_learning_rate))
        model.train_step.run(feed)
        if i == hps.train_steps - 1:
            cnn.save_model(sess, FLAGS.savePath, i)
    print('test accuracy %g' % model.accuracy.eval(feed_test))
    #feed = {x: testSet.data, y_: testSet.label, keep_prob: 1.0}


def trainer(model_params):
    """Train a cnn model."""
    # logging the hyperparams and loading the data set
    tf.logging.info('cnn')
    tf.logging.info('Hyperparams:')
    for key, val in model_params.values().iteritems():
        tf.logging.info('%s = %s', key, str(val))

    # load data
    tf.logging.info('Loading data files.')
    train_set, test_set, test_true_num, test_false_num = utils.read_data_set(FLAGS.dataSetPath, True)

    # build model
    reset_graph()  # Closes the current default session and resets the graph.
    model = cnn.Model(model_params)  # the model

    # run
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # # check if use the previous data
    # if FLAGS.resume_training:   # FLAGS.resume_training: Set to true to load previous checkpoint
    #     load_checkpoint(sess, FLAGS.log_root)

    train(sess, model, train_set, test_set, test_true_num, test_false_num)


def main(unused_argv):
    """Load model params, save config file and start trainer."""
    model_params = cnn.get_default_hparams()  # copy default hparams
    trainer(model_params)  # start train


def console_entry_point():
    tf.app.run(main)  # Runs the program with an optional 'main' function and 'argv' list.


if __name__ == '__main__':
    console_entry_point()

