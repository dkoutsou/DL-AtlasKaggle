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
        super(ResNetModel, self).init_build_model()

        logits = self.model(self.input_layer,
                            training=self.is_training)
        self.logits = tf.identity(logits, name="logits")

        super(ResNetModel, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
