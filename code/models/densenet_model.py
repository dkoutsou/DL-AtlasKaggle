import tensorflow as tf
from models.densenet_official import (conv, dense_block, transition_layer,
                                      _BATCH_NORM_DECAY, _BATCH_NORM_EPSILON)
from base.base_model import BaseModel

""" This file implements InceptionNet as a child
of our base model class. The function that actually
builds the model comes from the original Github of
Google (cf. models.inception_res_v2)
"""


class DenseNetModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        super(DenseNetModel, self).init_build_model()
        mapping = {
                121: [6, 12, 24, 16],
                169: [6, 12, 32, 32],
                201: [6, 12, 48, 32]
                }

        try:
            depths = mapping[self.config.densenet_size]
        except AttributeError:
            print('WARN: densenet_size not specified',
                  'using 121')
            depths = mapping[121]

        k = 32
        num_classes = 28

        num_channels = 2 * k
        v = conv(self.input_layer, filters=2 * k, strides=2, kernel_size=7)
        v = tf.layers.batch_normalization(
            inputs=v,
            axis=-1,
            training=self.is_training,
            fused=True,
            center=True,
            scale=True,
            momentum=_BATCH_NORM_DECAY,
            epsilon=_BATCH_NORM_EPSILON,
        )
        v = tf.nn.relu(v)
        v = tf.layers.max_pooling2d(v, pool_size=3, strides=2, padding="same")
        for i, depth in enumerate(depths):
            with tf.variable_scope("block-%d" % i):
                for j in range(depth):
                    with tf.variable_scope("denseblock-%d-%d" % (i, j)):
                        output = dense_block(v, k, self.is_training)
                        v = tf.concat([v, output], axis=3)
                        num_channels += k
                if i != len(depths) - 1:
                    num_channels /= 2
                    v = transition_layer(v, num_channels, self.is_training)

        global_pool = tf.reduce_mean(v, axis=(1, 2), name="global_pool")
        dense_layer = tf.layers.dense(global_pool, units=num_classes)
        self.logits = tf.identity(dense_layer, "logits")

        super(DenseNetModel, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
