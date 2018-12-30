import tensorflow as tf
from base.base_model import BaseModel
from models.inception_res_v2 import inception_resnet_v2

""" This file implements InceptionNet as a child
of our base model class. The function that actually
builds the model comes from the original Github of
Google (cf. models.inception_res_v2)
"""


class InceptionModel(BaseModel):
    def __init__(self, config):
        super(InceptionModel, self).__init__(config)
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
        x = tf.transpose(self.input, perm=[0, 3, 1, 2])

        logits, _ = inception_resnet_v2(x, num_classes=28,
                                        is_training=self.is_training,
                                        dropout_keep_prob=0.8,
                                        reuse=None,
                                        scope='InceptionResnetV2',
                                        create_aux_logits=True,
                                        activation_fn=tf.nn.relu)
        logits = tf.identity(logits, name="logits")
        # we have to adapt their code cause their code does
        # one label prediction, we want multilabel
        # use sigmoid not softmax because multilabel
        # then each out node is the proba the corresponding
        # label being true. I.e. if > 0.5 output the prediction.
        out = tf.nn.sigmoid(logits, name='out')
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
            self.prediction = tf.round(out, name="prediction")

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
