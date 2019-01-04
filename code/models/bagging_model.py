import tensorflow as tf
from base.base_model import BaseModel


class BaggingModel(BaseModel):
    def __init__(self, config, models):
        super(BaggingModel, self).__init__(config)
        self.build_model()
        self.init_saver()
        self.models = models

    def build_model(self):
        # The bagging model itself, does not build a graph
        # It's just a dummy model used for saving and loading functionality
        pass

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used
        # in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
