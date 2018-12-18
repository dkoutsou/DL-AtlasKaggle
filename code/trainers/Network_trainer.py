from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class NetworkTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(NetworkTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        iterator = self.data.batch_iterator()
        loop = tqdm(range(self.data.num_batches_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step(iterator)
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self, iterator):
        batch_x, batch_y = next(iterator)
        feed_dict = {
            self.model.input: batch_x,
            self.model.label: batch_y,
            self.model.is_training: True
        }
        _, loss, acc = self.sess.run([
            self.model.train_step, self.model.loss,
            self.model.accuracy
        ],
            feed_dict=feed_dict)
        return loss, acc
