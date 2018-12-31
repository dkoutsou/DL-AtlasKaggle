import argparse
import numpy as np


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def get_pred_from_probas(probas):
    tmp_pred = np.round(probas)
    for i in range(len(probas)):
        # if no classes predicted
        # i.e. no probas > 0.5
        # then choose the most probable one.
        if np.sum(tmp_pred[i]) == 0: 
            tmp_pred[i, np.argmax(probas[i])[0]] = 1
        # more than 3 classes predicted take the 3 most 
        # probable ones.
        elif np.sum(tmp_pred[i]) > 3:
            ind = np.argsort(probas[i])[-3:]
            tmp_pred[i] = np.zeros(28)
            tmp_pred[i, ind] = 1
    return(tmp_pred)
