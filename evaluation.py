from pylab import *
import numpy as np
from sklearn import metrics


def computeAcc(Labels, Ds, test_batch, test_interval, test_it, it, test_acc):
	correct = 0
	Predictions = np.zeros((Labels.size))

	Predictions[Ds < 1] = 1
	correct = correct + sum(Labels == Predictions)

	test_acc[it // test_interval] = float(correct) / float(test_batch * test_it)

	return test_acc


def PR_curve(scores, labels):
	n = len(scores)
	sort_idx = np.argsort(scores)
	sorted_scores = scores[sort_idx]
	sorted_labels = labels[sort_idx]

	tp = np.cumsum(sorted_labels)
	fp = np.cumsum(1 - sorted_labels)
	fn = np.sum(sorted_labels) - tp

	Precision = tp / (tp + fp)
	Recall = tp / (tp + fn)

	AUC = metrics.auc(Recall, Precision)

	return Recall, Precision, AUC
