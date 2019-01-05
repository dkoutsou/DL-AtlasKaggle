import tensorflow as tf

"""
Code inspired by https://github.com/zhezh/focalloss/blob/master/focalloss.py
Focal Loss from : Lin et al (2017).
                Focal Loss for Dense Object Detection, 130(4), 485–491.
                https://doi.org/10.1016/j.ajodo.2005.02.022
"""


def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
    Focal loss for multi-classification
    FL(p)=alpha(1-p)^{gamma}*(-ln(p))
    """

    # some classes can never exist in a batch,
    # so add a small value epsilon to softmax to stabilize the CE
    epsilon = 1.e-9

    logits = tf.convert_to_tensor(logits, tf.float32)

    sigmoid = tf.nn.sigmoid(logits)
    model_out = tf.add(sigmoid, epsilon)

    ce = tf.multiply(labels, -tf.log(model_out),
                     name='cross_entropy')
    weight = tf.multiply(labels,
                         tf.pow(tf.subtract(1., model_out),
                                gamma), name='fl_weight')
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    return fl


def f1_loss(y_true, y_pred):
    # y_pred should be the proba to make it differentiable
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, 'float'), axis=0)
    # tn = tf.reduce_sum(tf.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2*p*r / (p+r+tf.keras.backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(f1)
