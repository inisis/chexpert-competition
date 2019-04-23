import os
import sys
import argparse

import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# python bin/roc.py weights/dev_test.csv /data/ly/experiments/dev.csv weights/

parser = argparse.ArgumentParser(description='Plot ROC')

parser.add_argument('pred_csv_path', default=None, metavar='PRED_CSV_PATH',
                    type=str, help="Path to the prediction in csv")
parser.add_argument('true_csv_path', default=None, metavar='TRUE_CSV_PATH',
                    type=str, help="Path to the ground truth in csv")
parser.add_argument('plot_path', default=None, metavar='PLOT_PATH',
                    type=str, help="Path to the ROC plots")
parser.add_argument('--prob_thred', default=0.5, type=float,
                    help="Probability threshold")


def read_csv(csv_path):
    image_paths = []
    probs = []
    with open(csv_path) as f:
        header = f.readline().strip('\n').split(',')
        for line in f:
            fields = line.strip('\n').split(',')
            image_paths.append(fields[0])
            probs.append(list(map(float, fields[1:])))
    probs = np.array(probs)

    return (image_paths, probs, header)


def run(args):
    images_pred, probs_pred, header_pred = read_csv(args.pred_csv_path)
    images_true, probs_true, header_true = read_csv(args.true_csv_path)

    assert header_pred == header_true
    assert images_pred == images_true

    num_labels = len(header_true) - 1
    for i in range(num_labels):
        label = header_true[i+1]
        y_pred = probs_pred[:, i]
        y_true = probs_true[:, i]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(
            y_true, (y_pred >= args.prob_thred).astype(int), normalize=True
        )

        plt.figure(figsize=(8, 8), dpi=150)
        plt.xlim((0, 1.0))
        plt.ylim((0, 1.0))
        plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title('{} ROC, AUC : {:.3f}, Acc : {:.3f}'.format(label, auc, acc))
        plt.plot(fpr, tpr, '-b')
        plt.grid()
        plt.savefig(os.path.join(args.plot_path, label + '_roc.png'),
                    bbox_inches='tight')


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
