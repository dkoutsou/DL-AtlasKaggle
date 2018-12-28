import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE

# WIP
# Imblearn supports multi-class but not multi-output data
# (since sampling the minority class means we are sampling
# other classes too)
# TODO:
# - Iteratively sample from minority class ignoring label relations
#   and setting rest of labels of generated samples to 0. (i.e
#   generated samples belong ONLY to minority class)
# - Build label relation graph, partition label sapce based on graph,
#   oversample and then inverse transform back


class Oversampler:
    """ This class is a wrapper around
    imblearn.oversample.
    """

    def __init__(self, type='adasyn'):
        """
        Args:
            type: Which algo to use: 'adasyn' or 'smote'
        """
        if type not in ['adasyn', 'smote']:
            print("ERR: Unrecognized method: {}".format(type))
        self.type = type

    def _init_sampler(self, sampling_strategy='auto', neighbors=5):
        sampler = None
        if self.type == 'adasyn':
            sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=neighbors,
                n_jobs=4)
        elif self.type == 'smote':
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=neighbors,
                n_jobs=4)
        return sampler

    def _generate_by_class(self, X, y, class_id, num_of_samples):
        # pick only column class_id
        y_temp = y[:, class_id]

        # in a single column there are two classes: 0 and 1
        # generate samples for class 1
        sampler = self._init_sampler({1: num_of_samples})
        Xnew, _ = sampler.fit_resample(X, y_temp)

        # new samples are always added to the end so pick last num_of_samples
        Xnew = Xnew[-num_of_samples:, :]

        # create labels for generated samples
        y_new = np.zeros((num_of_samples, y.shape[1]))
        y_new[:, class_id] = 1

        return Xnew, y_new

    def resample_naive(self, X, y, step=200, imb_ratio=20):
        """
        Iteratively adds samples to the minority class until the
        given imbalance ratio is reached. Imbalance ratio of 1
        means there is no imbalance.
        Args:
            step: how many samples to add to the minority class
              in each iteration
            imb_ratio: majority/minority. The algorithm stops
              once this ratio is reached.
        """
        Xresampled = X.copy()
        yresampled = y.copy()
        counts = np.sum(y, axis=0)
        ratio = np.max(counts) / np.min(counts)
        while ratio > imb_ratio:
            print("Imbalance ratio:", ratio)
            minority_class = np.argmin(counts)

            # input original X and y (not resampled ones)
            X_tmp, y_tmp = self._generate_by_class(X, y, minority_class, step)
            # print(X_tmp, y_tmp)
            Xresampled = np.vstack((Xresampled, X_tmp))
            yresampled = np.vstack((yresampled, y_tmp))

            # calculate imbalance ratio using resampled matrices
            counts = np.sum(yresampled, axis=0)
            ratio = np.max(counts) / np.min(counts)
        print("Imbalance ratio:", ratio)
        return Xresampled, yresampled

    def resample(self, X, y, strategy='naive'):
        """ Oversamples data and returns updated
        data X(samples*features) and target labels y

        Args:
            X: samples * features
            y: target labels y
        """
        if strategy == 'naive':
            return self.resample_naive(X, y)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    X1 = np.random.normal(size=10000).reshape(-1, 1)
    y1 = np.zeros((10000, 3))
    y1[:, 0] = 1

    X2 = np.random.normal(size=60).reshape(-1, 1) + 1.5
    y2 = np.zeros((60, 3))
    y2[:, 1] = 1

    X3 = np.random.normal(size=30).reshape(-1, 1) + 2.5
    y3 = np.zeros((30, 3))
    y3[:, 2] = 1

    X = np.vstack((X1, X2, X3))
    y = np.vstack((y1, y2, y3))

    sampler = Oversampler()
    print(X.shape, y.shape)
    Xnew, ynew = sampler.resample(X, y)
    print(Xnew.shape, ynew.shape)
    print(Xnew, ynew)
