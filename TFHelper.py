import numpy as np
import tensorflow as tf
import functools
import sys

# create tf weight and biases var pair
def var(kernel_shape):
    """weights biases var init"""
    weight = tf.get_variable("weights", kernel_shape,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("biases", [kernel_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    return weight, biases


# create Conv Layer Model
def conv_relu(x_input, kernel_shape, pool=False, drop=None):
    """build conv relu layer"""
    weights, biases = var(kernel_shape)
    conv = tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME')
    print(conv)
    rtn = tf.nn.relu(conv + biases)
    if pool:
        rtn = tf.nn.max_pool(rtn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if drop is not None and drop != 0.:
        rtn = tf.nn.dropout(rtn, drop)
    print(rtn)
    return rtn


# relu layer
def relu(x_input, kernel_shape, drop=None):
    """build relu layer"""
    weights, biases = var(kernel_shape)
    rtn = tf.nn.relu(tf.matmul(x_input, weights) + biases)
    if drop is not None and drop != 0.:
        rtn = tf.nn.dropout(rtn, drop)
    print(rtn)
    return rtn


# 打印步骤信息
def step_info(step, total_step, scores):
    sys.stdout.write('\r--== Step: ' + str(step) + '/' + str(total_step) +
                     ' (' + str(round(step / total_step * 100, 1)) + '%)' +
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

    def fit_generator(self, train_generator, vail_data, vail_labs):
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
            tf_train_shaped = tf.reshape(self.tf_train_data,
                                         shape=[-1, image_height, image_width, num_channels])
            print(self.tf_train_data)
            tf_train_labs = tf.placeholder(tf.float32, shape=(batch_size, label_len))
            tf_vail_data = tf.constant(vail_data)

            # Training computation.
            with tf.variable_scope("predict") as scope:
                logits, self.train_predict = self.func_model(tf_train_shaped, drop=self.tf_drop)
                print(logits)
                scope.reuse_variables()
                _, vail_prediction = self.func_model(tf_vail_data, drop=None)

            loss = self.func_loss(logits, tf_train_labs)
            # Optimizer.
            decay_lr = tf.train.exponential_decay(self.learning_rate, global_step,
                                                  self.steps, 0.5, staircase=True)
            optimizer = self.func_optimizer(decay_lr).minimize(loss)

        #########################

        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            print('Initialized')
            for step in range(self.steps):
                feed_dict = {
                    self.tf_train_data: x,
                    tf_train_labs: y.reshape(batch_size, label_len),
                    self.tf_drop: self.drop,
                }
                _, l, train_p, vail_p = session.run(
                    [optimizer, loss, self.train_predict, vail_prediction], feed_dict=feed_dict)

                if step % 50 == 0:
                    acck, accv = self.func_accuracy(train_p, y)
                    acctk, acctv = self.func_accuracy(vail_p, vail_labs)
                    step_info(step, self.steps, {
                        'loss': l,
                        'train ' + acck: accv,
                        'vail ' + acctk: acctv,
                    })
                    saver.save(session, './' + self.filename)
                    if step % 200 == 0 and step != 0:
                        print("")
                x, y = next(train_generator)

            print("\nTest %s: %.1f" % self.func_accuracy(vail_p, vail_labs))

    def predict(self, x_data):
        with tf.Session(graph=self.graph) as session:
            saver = tf.train.Saver()
            saver.restore(session, './' + self.filename)
            feed_dict = {
                self.tf_train_data: x_data,
                self.tf_drop: 1,
            }
            return session.run(self.train_predict, feed_dict=feed_dict)


