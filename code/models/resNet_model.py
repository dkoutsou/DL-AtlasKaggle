import tensorflow as tf
from base.base_model import BaseModel
from models.resnet_official import Model, _get_block_sizes

""" This file implements InceptionNet as a child
of our base model class. The function that actually
builds the model comes from the original Github of
Google (cf. models.inception_res_v2)
"""


class ResNetModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        # For bigger models, we want to use "bottleneck" layers
        try:
            if self.config.resnet_size < 50:
                bottleneck = False
            else:
                bottleneck = True
        except AttributeError:
            # sizes allowed: dict_keys([18, 34, 50, 101, 152, 200])
            print('WARN: resnet_size not specified',
                  'using 101')
            self.config.resnet_size = 101
            bottleneck = True
        self.model = Model(resnet_size=self.config.resnet_size,
                           bottleneck=bottleneck,
                           num_classes=28,
                           num_filters=64,
                           kernel_size=7,
                           conv_stride=2,
                           first_pool_size=3,
                           first_pool_stride=2,
                           block_sizes=_get_block_sizes(
                               self.config.resnet_size),
                           block_strides=[1, 2, 2, 2],
                           resnet_version=2,
                           data_format=None,
                           dtype=tf.float32)

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

        logits = self.model(self.input, training=self.is_training)
        logits = tf.identity(logits, name="logits")
        # we have to adapt their code cause their code does
        # one label prediction, we want multilabel
        # use sigmoid not softmax because multilabel
        # then each out node is the proba the corresponding
        # label being true. I.e. if > 0.5 output the prediction.
        with tf.name_scope("loss"):
            if self.config.use_weighted_loss:
                tf.stop_gradient(self.class_weights, name="stop_gradient")
                self.loss = tf.losses.compute_weighted_loss(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=logits),
                    weights=self.class_weights)
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.label, logits=logits))
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(
                    self.loss, global_step=self.global_step_tensor)
        with tf.name_scope("output"):
            self.out = tf.nn.sigmoid(logits, name='out')
            self.prediction = tf.round(self.out, name="prediction")

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
