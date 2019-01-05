from base.base_train import BaseTrain
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from trainers.Network_trainer import NetworkTrainer
from utils.utils import get_pred_from_probas


class BaggingTrainer(BaseTrain):
    def __init__(self, sessions, models, data_gens, model_configs, loggers,
                 master_sess, master_model, master_config, master_logger):
        super(BaggingTrainer, self).__init__(master_sess, master_model, None,
                                             master_config, master_logger)
        self.sessions = sessions
        self.models = models
        self.data_gens = data_gens
        self.model_configs = model_configs
        self.loggers = loggers
        self.trainers = []
        for sess, model, data, config, logger in zip(
                self.sessions, self.models, self.data_gens, self.model_configs,
                self.loggers):
            self.trainers.append(
                NetworkTrainer(sess, model, data, config, logger))

    def train_epoch(self):
        n_estimators = self.config.n_estimators
        for i in range(
                self.model.cur_estimator_tensor.eval(self.sess), n_estimators,
                1):
            print("Training estimator {}".format(i))
            self.trainers[i].train_epoch()
            self.sess.run(self.model.increment_cur_estimator_tensor)
            if self.model.cur_estimator_tensor.eval(self.sess) == n_estimators:
                self.sess.run(self.model.reset_cur_estimator_tensor)

        # Evaluate on val every epoch
        cur_it = self.models[0].global_step_tensor.eval(self.sessions[0])
        self.sess.run(tf.assign(self.model.global_step_tensor, cur_it))
        val_loss, val_f1 = self.val_step()
        epoch = self.model.cur_epoch_tensor.eval(self.sess)
        print('Epoch {}: val_loss:{}, val_f1:{}'.format(
            epoch, val_loss, val_f1))
        val_summaries_dict = {'loss': val_loss, 'f1': val_f1}
        self.logger.summarize(
            cur_it, summaries_dict=val_summaries_dict, summarizer='test')

        self.model.save(self.sess)

    def val_step(self):
        val_iterator = self.data_gens[0].batch_iterator(type='val')
        val_losses = []
        val_probas = []
        val_true = []
        for batch_x, batch_y in val_iterator:
            probas = []
            for i in range(len(self.models)):
                m = self.models[i]
                feed_dict = {
                    m.input: batch_x,
                    m.label: batch_y,
                    m.is_training: False,
                    m.class_weights: self.data_gens[0].class_weights
                }
                loss, out = self.sessions[i].run([m.loss, m.out],
                                                 feed_dict=feed_dict)
                val_losses.append(loss)
                probas.append(out)
            probas_acc = np.zeros(probas[0].shape)
            for p in probas:
                probas_acc += p
            probas_acc = probas_acc / len(self.models)
            val_probas = np.append(val_probas, probas_acc)
            val_true = np.append(val_true, batch_y)
        val_true = np.reshape(val_true, (-1, 28))
        val_probas = np.reshape(val_probas, (-1, 28))
        val_preds = get_pred_from_probas(val_probas)
        val_f1 = f1_score(val_true, val_preds, average='macro')
        val_loss = np.mean(val_losses)
        return val_loss, val_f1
