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
        super(InceptionModel, self).init_build_model()

        logits, _ = inception_resnet_v2(self.input_layer,
                                        num_classes=28,
                                        is_training=self.is_training,
                                        dropout_keep_prob=0.8,
                                        reuse=None,
                                        scope='InceptionResnetV2',
                                        create_aux_logits=True,
                                        activation_fn=tf.nn.relu)
        self.logits = tf.identity(logits, name="logits")

        super(InceptionModel, self).build_loss_output()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
