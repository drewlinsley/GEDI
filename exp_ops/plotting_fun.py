import re
import itertools
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics


def acc_index(accs, ckpt_names):
    out = []
    for idx, acc in enumerate(accs):
        out = np.append(out, np.repeat(idx, len(acc)))
    return out


def precision_recall(y, score):
    p, r, _ = metrics.precision_recall_curve(y, score)
    mu_precision = metrics.average_precision_score(y, score)
    return p, r, mu_precision


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def plot_accuracies(y, yhat, config, ckpt_names, output_file):
    accs = (np.hstack((y[:])) == np.hstack((yhat[:]))).astype(np.float32)
    x_labs = acc_index(y, ckpt_names).astype(int) + 1
    x_labs = config.epochs - np.max(x_labs) + x_labs
    df = pd.DataFrame(np.vstack((accs * 100, x_labs)).transpose(),
        columns=['Validation image batch accuracies', 'Training iteration'])
    plt.figure(figsize=(30, 8)).add_subplot(1, 1, 1)
    sns.set(style="whitegrid")
    ax = sns.factorplot(x="Training iteration", y="Validation image batch accuracies", data=df,
        palette='GnBu_d', size=10, aspect=1.5, ci=95)
    plt.ylabel('Validation image batch classification accuracy (%)')
    plt.xlabel('Model training iteration')
    ax.set_xticklabels(ckpt_names, rotation=60)
    ax.set(ylim=(50, 105))
    plt.savefig(output_file)
    plt.close('all')


def plot_cost(cost_file, ckpt_names, output_file):
    costs = np.load(cost_file)
    smoothed_cost = movingaverage(costs, 200)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(smoothed_cost)
    ax.set_ylabel('Smoothed cost')
    ax.set_xlabel('Model training iteration')
    ax.set_xticklabels(ckpt_names)
    plt.savefig(output_file)


def plot_std(y, yhat, ckpt_names, output_file):
    accs = (np.hstack((y[:])) == np.hstack((yhat[:]))).astype(np.float32)
    std_accs = np.std(accs * 100, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(std_accs)
    ax.set_ylabel('Validation accuracy (%) standard deviation')
    ax.set_xlabel('Model training iteration')
    ax.set_xticklabels(ckpt_names)
    plt.savefig(output_file)


def plot_pr(ckpt_y, ckpt_yhat, ckpt_scores, output_file):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    mu_accs = np.asarray([np.mean(ckpt_y[idx] == ckpt_yhat[idx])
        for idx in range(len(ckpt_y))])
    argmax_acc = np.argmax(mu_accs)
    max_acc = np.max(mu_accs)
    p, r, mu_p = precision_recall(enc.fit_transform(ckpt_y[argmax_acc].reshape(-1, 1))
        .todense().reshape(-1, 1), ckpt_scores[argmax_acc].reshape(-1, 1))
    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(r, p, lw=2, label='Precision-Recall curve for checkpoint ' +
        str(argmax_acc) + '; ' + str(max_acc * 100) + '%')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(mu_p))
    plt.legend(loc="lower left")
    plt.savefig(output_file)


def plot_cms(ckpt_y, ckpt_yhat, config, output_file ,normalize=True):
    mu_accs = np.asarray([np.mean(ckpt_y[idx] == ckpt_yhat[idx])
        for idx in range(len(ckpt_y))])
    argmax_acc = np.argmax(mu_accs)
    max_acc = np.max(mu_accs)
    y = ckpt_y[argmax_acc]
    yhat = ckpt_yhat[argmax_acc]
    classes = [re.split('/', x)[-1] for x in config.raw_im_dirs]
    cm = metrics.confusion_matrix(y, yhat)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.grid(False)
    plt.title(config.which_dataset + ' confusion matrix; ' + str(max_acc * 100)
        + '%')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file)
