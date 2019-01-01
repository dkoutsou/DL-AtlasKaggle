import tensorflow as tf
from base.base_model import BaseModel
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from loss import focal_loss
from tensorflow.python.ops import math_ops



class CP4Model(BaseModel):
    def __init__(self, config):
        super(CP4Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        try:
            if self.config.use_weighted_loss:
                pass
        except AttributeError:
            print('WARN: use_weighted_loss not set - using False')
            self.config.use_weighted_loss = False
        self.is_training = tf.placeholder(tf.bool)
        self.class_weights = tf.placeholder(
            tf.float32, shape=[1, 28], name="weights")
        self.input = tf.placeholder(
            tf.float32, shape=[None, 4, 512, 512], name="input")
        self.label = tf.placeholder(tf.float32, shape=[None, 28])

        # All tf functions work better with channel first
        # otherwise some fail on CPU (known issue)
        x = tf.transpose(self.input, perm=[0, 2, 3, 1])
        # Block 1
        x = tf.layers.conv2d(x, 64, 3, padding='same', name='conv1')
        x = tf.nn.relu(x, name='act1')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool1')
        # Block 2
        x = tf.layers.conv2d(x, 128, 3, padding='same', name='conv2')
        x = tf.nn.relu(x, name='act2')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool2')
        # Block 3
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv3')
        x = tf.nn.relu(x, name='act3')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool3')
        # Block 4
        x = tf.layers.conv2d(x, 256, 3, padding='same', name='conv4')
        x = tf.nn.relu(x, name='act4')
        x = tf.layers.max_pooling2d(
            x, pool_size=(2, 2), strides=(2, 2), name='pool4')
        # Classification block
        x = tf.layers.flatten(x, name='flatten')
        x = tf.nn.relu(x, name='act5')
        logits = tf.layers.dense(x, units=28, name='logits')
        out = tf.nn.softmax(logits, name='out')
        with tf.name_scope("loss"):
            if self.config.focalLoss:
                print("Using focal loss")
                self.loss = tf.losses.compute_weighted_loss(
                    focal_loss(labels=self.label, logits=logits, gamma=2), weights=self.class_weights)
            elif self.config.use_weighted_loss_1:
                tf.stop_gradient(self.class_weights, name="stop_gradient")
                self.loss = tf.losses.compute_weighted_loss(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=logits),
                    weights=self.class_weights)
            elif self.config.use_weighted_loss_2:
                self.loss = tf.nn.weighted_cross_entropy_with_logits(
                     targets=self.label, logits=logits, pos_weight=self.class_weights
                )
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=logits))
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)
        with tf.name_scope("output"):
            self.prediction = tf.round(out, name="prediction")

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

# adapted from https://github.com/zhezh/focalloss
def focalLoss(labels, logits, gamma=2, alpha=4, name=None):
    print(tf.__version__)

    #with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
    #    logits = ops.convert_to_tensor(logits, name="logits")
    #    labels = ops.convert_to_tensor(tf.to_int64(labels), name="labels")
    #    try:
    #        labels.get_shape().merge_with(logits.get_shape())
    #    except ValueError:
    #        raise ValueError("logits and labels must have the same shape (%s vs %s)" %
    #                         (logits.get_shape(), labels.get_shape()))

    #zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    #cond = (logits >= zeros)
    #relu_logits = array_ops.where(cond, logits, zeros)
    #neg_abs_logits = array_ops.where(cond, -logits, logits)


    print('logist: {}'.format(logits.dtype))
    print('labels: {}'.format(labels.dtype))
    print(labels)
    labels = tf.convert_to_tensor(tf.to_int64(labels), name="labels")
    logits = tf.convert_to_tensor(logits, tf.float32, name="logits")
    num_class = logits.shape[1]
    print('num_cls: {}'.format(num_class))

    onehot_labels = tf.one_hot(labels, num_class)
    print("one hot shape: {}".format(tf.shape(onehot_labels)))
    ce = tf.multiply(onehot_labels, -tf.log(logits))  # -log(p)
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., logits), gamma))  # (1-p)^gamma
    print('ce: {}'.format(ce.dtype))
    print('weight: {}'.format(weight.dtype))
    print('ce: {}'.format(tf.shape(ce)))
    print('weight: {}'.format(tf.shape(weight)))
    print('weight*ce: {}'.format(tf.shape(tf.multiply(weight, ce))))
    ce = -tf.log(logits)
    weight = tf.pow(tf.subtract(1., logits), gamma)
    focal_loss = tf.multiply(tf.to_float(alpha), tf.multiply(ce, weight))
    return tf.reduce_sum(focal_loss, axis=1)



    #if not (target.size() == input.size()):
    #    raise ValueError("Target size ({}) must be the same as input size ({})"
    #                     .format(target.size(), input.size()))

    #max_val = tf.clip_by_value((-input), clip_value_min=0, clip_value_max=(-input))
    #loss = input - input * target + max_val + (tf.exp(-max_val) + tf.log(tf.exp((-input - max_val))))

    #invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    #loss = tf.exp((invprobs * gamma)) * loss
    #print(loss)

    #return tf.reduce_sum(loss)


def focalLoss_tf(labels, logits, gamma=2, alpha=4, name=None):

    print(type(input))
    print(tf.__version__)

    with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        labels = ops.convert_to_tensor(tf.to_int64(labels), name="labels")
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (logits.get_shape(), labels.get_shape()))

    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = array_ops.where(cond, logits, zeros)
    neg_abs_logits = array_ops.where(cond, -logits, logits)
    return math_ops.add(
        relu_logits - logits * labels,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=name)

