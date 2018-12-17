import tensorflow as tf

from data_loader.data_generator import DataGenerator, DataTestLoader
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import parmap
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MultiLabelBinarizer


def get_features_from_batch_images(img, r, p):
    """ Get the features from one image.
    Global intensity histogram per channel
    and local histogram on image patches.

    Args:
        img: the image with 4 input channel
        r: 512//p number of patches in 
            one side of the image
        p: patch size for local histogram
    """
    tmp_feats = []
    for channel in range(4):
        current_img = img[channel, :, :]
        tmp_feats = np.append(tmp_feats, np.histogram(current_img)[0])
        # extract 8*8 patches of 64*64 px and derive 10 bins histogram
        for j in range(r):
            for k in range(r):
                tmp_feats = np.append(tmp_feats, np.histogram(
                    current_img[j*p:(j+1)*(p), k*p:(k+1)*p])[0])
    return tmp_feats


def extract_features(all_batches, config, train=True):
    """ Main features extraction function.

    Args:
        all_batches: batches iterator either from 
                    DataGenerator class or from TestLoader
                    class. If TestLoader set train to False.
        config: config file
        train: boolean to specify with DataLoader is used and
                whether should return labels or not.
    Returns:
        if train:
            feats: [n_samples, n_feats]
            labels: [n_samples, 1]
        else:
            feats: [n_samples, n_feats]
    """
    # manually derive basic intensities features
    # takes 20 sec / 1048 images batch on my laptop in 4 cores //
    counter = 1
    p = config.patch_size
    r = 512//p
    if train:
        for batch_img, batch_label in all_batches:
            # just for testing just use 20 batch as training set
            #if counter > 20:
            #    break
            print('processing batch {}'.format(counter))
            if counter == 1:
                labels = batch_label
            else:
                labels = np.concatenate((labels, batch_label))
            t1 = time.time()
            feats = np.asarray(parmap.map(
                get_features_from_batch_images, batch_img, r, p, pm_pbar=True))
            print(time.time()-t1)
            counter += 1
        return feats, labels
    else:
        for batch_img in all_batches:
            # just for testing just use 20 batch as training set
            #if counter > 20:
            #    break
            print('processing batch {}'.format(counter))
            t1 = time.time()
            feats = np.asarray(parmap.map(
                get_features_from_batch_images, batch_img, r, p, pm_pbar=True))
            print(time.time()-t1)
            counter += 1
        return feats


def get_baseline_CV_score(feats, labels, estimator, scores=['f1_macro']):
    """ Calculate the cross-validation score
    of the baseline estimator

    Args:
        feats: input features to classifiers
        labels: one-hot multilabel representation
        scores: array of scoring function to assess by CV

    Returns:
        cv_scores: an array of scores objects
    """
    print(np.shape(labels))
    feats = np.reshape(feats, (len(labels), -1))
    print(np.shape(feats))
    print(np.shape(labels))
    cv_scores = []
    for score in scores:
        cv_scores = np.append(cv_scores, cross_val_score(
            estimator, feats, labels, scoring=score))
    return cv_scores


def fit_predict(train_feats, train_labels, test_feats, estimator):
    """ Wrapper for the fit + predict pipeline. 

    Args:
        train_feats: matrix[n_samples, n_feats] with training features
        train_labels: one-hot multilabel representation [n_samples, n_classes]
        estimator: object derived from Sklearn BaseEstimator

    Note:
        RF estimator can return None as class, don't know how Kaggle
        handles that. TODO: check this case.
    """
    bin = MultiLabelBinarizer(classes=np.arange(28))
    bin.fit(train_labels)  # needed for instantiation of the object
    estimator.fit(train_feats, train_labels)
    one_hot_pred = estimator.predict(test_feats)
    predicted_labels = bin.inverse_transform(one_hot_pred)
    return(predicted_labels)


def main():
    """ Main procedure: extract features,  
    get RF cross-validation performance, 
    fit and predict baseline and save csv for Kaggle.

    Note:
        Set the batch_size to some big number ~1000
        for better parmap performance.
    """
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except Exception:
        print("missing or invalid arguments")
        exit(0)

    # create your data generator
    TrainingSet = DataGenerator(config)
    all_batches = TrainingSet.batch_iterator()
    # extract features
    train_feats, train_labels = extract_features(all_batches, config)

    # get cv score
    rf = RandomForestClassifier(n_estimators=100)
    cv_scores = get_baseline_CV_score(train_feats, train_labels, rf)
    print(cv_scores)

    # Load Test Set
    TestSet = DataTestLoader(config)
    test_batches = TestSet.batch_iterator()

    # Fit and predict for Kaggle
    test_feats = extract_features(test_batches, config, train=False)
    prediction = fit_predict(train_feats, train_labels, test_feats, rf)
    ids = TestSet.image_ids
    result = pd.DataFrame()
    
    l = [' '.join([str(p) for p in sample_pred]) for sample_pred in prediction]
    result['Id'] = ids[0:len(l)]
    result['Predict'] = l
    print(result)
    create_dirs([config.result_folder+config.exp_name])
    result.to_csv(config.result_folder+config.exp_name+'/prediction.csv', index=False)


if __name__ == '__main__':
    main()
