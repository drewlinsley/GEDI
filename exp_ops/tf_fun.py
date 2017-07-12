import os
import numpy as np
import tensorflow as tf
from glob import glob


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def fine_tune_prepare_layers(tf_vars, finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        ss = [v.name.find(x) != -1 for x in finetune_vars]
        if True in ss:
            ft_vars.append(v)
        else:
            other_vars.append(v)
    return other_vars, ft_vars


def ft_optimized(cost, var_list_1, var_list_2, optimizer, lr_1, lr_2):  # applies different learning rates to specified layers
    opt1 = optimizer(lr_1)
    opt2 = optimizer(lr_2)
    grads = tf.gradients(cost, var_list_1 + var_list_2)
    grads1 = grads[:len(var_list_1)]
    grads2 = grads[len(var_list_1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list_1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list_2))
    return tf.group(train_op1, train_op2)


def ft_non_optimized(cost, other_opt_vars, ft_opt_vars, optimizer, lr_1, lr_2):
    op1 = optimizer(lr_1).minimize(cost, var_list=other_opt_vars)
    op2 = optimizer(lr_2).minimize(cost, var_list=ft_opt_vars)
    return tf.group(op1, op2)  # ft_optimize is more efficient. siwtch to this once things work


def class_accuracy(pred, targets):
    return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(pred, 1),
        tf.cast(targets, dtype=tf.int64))))  # assuming targets is an index


def tf_confusion_matrix(pred, targets):
    return tf.contrib.metrics.confusion_matrix(pred, targets)  # confusion


def softmax_cost(logits, labels, ratio=None, rebalance=1, flip_ratio=True):
    if ratio is not None:
        if rebalance is not None:
            ratio *= rebalance
        if flip_ratio:
            ratio = ratio[::-1]
        print 'Reweighting logits to: %s' % ratio

        ratios = tf.get_variable(
            name='ratio', initializer=ratio)[None, :]
        weights_per_label = tf.matmul(
            tf.one_hot(labels, int(logits.get_shape()[-1])), tf.transpose(tf.cast(ratios, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)))
    else:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels))


def find_ckpts(config,dirs=None):
    if dirs is None:
        dirs = sorted(glob(
        config.train_checkpoint + config.which_dataset + '*'), reverse=True)[0]  # only the newest model run
    ckpts = sorted(glob(dirs + '/*.ckpt*'))
    ckpts = [x for x in glob(os.path.join(dirs, '*.ckpt*')) if 'meta' in x]
    ckpt_num = np.argsort([int(x.split('-')[-1].split('.')[0]) for x in ckpts])
    ckpt_metas = np.asarray(ckpts)[ckpt_num]
    ckpt_names = [x.split('.meta')[0] for x in ckpt_metas]
    return np.asarray(ckpt_names), ckpt_metas
