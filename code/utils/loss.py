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
