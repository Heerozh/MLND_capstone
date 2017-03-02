import numpy as np
import tensorflow as tf
import functools
import sys
from IPython.display import display, HTML
import time

TREE = []
def add_display_tree(net, add=''):
    TREE.append((net.name + ' ' + str(add), str(net.get_shape().as_list()[1:])))
def print_tree():
    html = ["<table width=50%>"]
    for row in TREE:
        html.append("<tr>")
        html.append("<td>{0}</td> <td>{1}</td>".format(*row))
        html.append("</tr>")
    html.append("</table>")
    display(HTML(''.join(html)))


class Layer:
    stddev = 0.08
    # create tf weight and biases var pair
    @staticmethod
    def var(kernel_shape):
        """weights biases var init"""
        weight = tf.get_variable("weights", kernel_shape,
                                 initializer=tf.truncated_normal_initializer(stddev=Layer.stddev))
        biases = tf.get_variable("biases", [kernel_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        return weight, biases

    # create Conv Layer Model
    @staticmethod
    def conv_relu(x_input, kernel_shape, stride=1, padding='SAME'):
        """build conv relu layer"""
        weights, biases = Layer.var(kernel_shape)
        conv = tf.nn.conv2d(x_input, weights, strides=[1, stride, stride, 1], padding=padding)
        add_display_tree(conv, str(kernel_shape[:2]) +
                         (stride > 1 and ' /' + str(stride) or ''))
        rtn = tf.nn.relu(conv + biases)
        return rtn

    @staticmethod
    def pool(x_input, ksize=2):
        rtn = tf.nn.max_pool(x_input, ksize=[1, ksize, ksize, 1], strides=[1, 2, 2, 1], padding='SAME')
        add_display_tree(rtn, '/2')
        return rtn

    # relu layer
    @staticmethod
    def relu(x_input, out_size):
        """build relu layer"""
        shape = x_input.get_shape().as_list()
        weights, biases = Layer.var([shape[1], out_size])
        rtn = tf.nn.relu(tf.matmul(x_input, weights) + biases)
        add_display_tree(rtn)
        return rtn

    @staticmethod
    def drop(x_input, drop):
        if drop is not None and drop != 0.:
            rtn = tf.nn.dropout(x_input, drop)
            add_display_tree(rtn)
            return rtn
        return x_input

    @staticmethod
    def flat(x_input):
        shape = x_input.get_shape().as_list()
        size = shape[1] * shape[2] * shape[3]
        reshape = tf.reshape(x_input, [-1, size])
        add_display_tree(reshape)
        return reshape

    @staticmethod
    def avg_pool(x_input):
        rtn = tf.nn.avg_pool(x_input,
                             ksize=[1, x_input.get_shape().as_list()[1],
                                    x_input.get_shape().as_list()[2], 1],
                             strides=[1, 1, 1, 1], padding='VALID')
        add_display_tree(rtn)
        return rtn

    @staticmethod
    def split():
        TREE.append(('---', '---'))


# 打印步骤信息
def step_info(step, total_step, scores, starttime):
    sys.stdout.write('\r--== Step: ' + str(step) + '/' + str(total_step) +
                     ' (' + str(round(step / total_step * 100, 1)) + '%) ' +
                     str(round(time.time() - starttime)) + 's' +
                     ' Score: ' + ', '.join("{!s} {!s}".format(key, val)
                                            for (key, val) in scores.items()) +
                     ' ==-- ')


# ts pack
class Learner:
    graph = None
    train_predict = None
    tf_drop = None
    tf_train_data = None

    def __init__(self, model, accuracy, filename,
                 steps=1001, learning_rate=0.1,
                 loss=tf.nn.sigmoid_cross_entropy_with_logits,
                 optimizer=tf.train.AdamOptimizer,
                 drop=0.85
                ):
        self.func_model = model
        self.func_accuracy = accuracy
        self.steps = steps
        self.func_loss = loss
        self.func_optimizer = optimizer
        self.learning_rate = learning_rate
        self.drop = drop
        self.filename = filename

    def fit_generator(self, train_generator, vail_data, vail_labs, restore=False):
        x, y = next(train_generator)

        print(x.shape)
        batch_size = x.shape[0]
        image_height = x.shape[1]
        image_width = x.shape[2]
        num_channels = x.shape[3]
        label_len = functools.reduce(np.dot, y.shape[1:])

        self.graph = tf.Graph()
        with self.graph.as_default():
            global_step = tf.Variable(0, trainable=False)
            # Input data.
            self.tf_drop = tf.placeholder_with_default(tf.constant(0.), None)
            self.tf_train_data = tf.placeholder(tf.float32)
            print(self.tf_drop)
            print(self.tf_train_data)
            tf_train_shaped = tf.reshape(self.tf_train_data,
                                         shape=[-1, image_height, image_width, num_channels])
            add_display_tree(tf_train_shaped, 'input')

            tf_train_labs = tf.placeholder(tf.float32, shape=(batch_size, label_len))
            tf_vail_data = tf.constant(vail_data)

            # Training computation.
            with tf.variable_scope("model") as scope:
                model = self.func_model(tf_train_shaped, drop=self.tf_drop)
                logits = model['logits']
                self.train_predict = model['predict']
                print(self.train_predict)
                add_display_tree(logits)
                print_tree()
                scope.reuse_variables()
                vail_prediction = self.func_model(tf_vail_data, drop=None)['predict']

            loss = self.func_loss(logits, tf_train_labs)
            # Optimizer.
            if self.learning_rate is None:
                optimizer = self.func_optimizer().minimize(loss)
            else:
                decay_lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                                      self.steps, 0.2, staircase=True)
                optimizer = self.func_optimizer(decay_lr).minimize(loss)

        #########################

        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            if restore:
                try:
                    saver.restore(session, './' + self.filename)
                except Exception as e:
                    print('cannot restore model, start over: ', e)
                    pass
            print('Initialized')
            start = time.time()
            for step in range(self.steps):
                feed_dict = {
                    self.tf_train_data: x,
                    tf_train_labs: y.reshape(batch_size, label_len),
                    self.tf_drop: self.drop,
                    global_step: step,
                }

                if step % 50 == 0 or step == (self.steps - 1):
                    _, l, train_p, vail_p = session.run(
                        [optimizer, loss, self.train_predict, vail_prediction],
                        feed_dict=feed_dict)
                    acck, accv = self.func_accuracy(train_p, y)
                    acctk, acctv = self.func_accuracy(vail_p, vail_labs)
                    step_info(step, self.steps, {
                        'loss': l,
                        'train ' + acck: accv,
                        'vail ' + acctk: acctv,
                    }, start)
                    start = time.time()
                    saver.save(session, './' + self.filename)
                else:
                    session.run(optimizer, feed_dict=feed_dict)
                x, y = next(train_generator)

    def predict(self, x_data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            saver.restore(session, './' + self.filename)
            feed_dict = {
                self.tf_train_data: x_data,
                self.tf_drop: 1,
            }
            return session.run(self.train_predict, feed_dict=feed_dict)


def load_and_predict( x_data, returnvar, filename):
    graph = tf.Graph()
    with graph.as_default():
        # with tf.device("/gpu:0"):
        saver = tf.train.import_meta_graph('./' + filename + ".meta")
    with tf.Session(graph=graph) as session:
        # with tf.device('/gpu:0'):
        saver.restore(session, './' + filename)
        # tf.global_variables_initializer().run()
        feed_dict = {
            'Placeholder:0': x_data,
            'PlaceholderWithDefault:0': 1,
        }
        return session.run(returnvar, feed_dict=feed_dict)


