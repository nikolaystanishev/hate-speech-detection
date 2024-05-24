import os
import re
import sys
import matplotlib.pyplot as plt

def get_empty_metrics():
    return {
        "loss": [],
        "macro_f1": [],
        "micro_f1": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "roc_auc": []
    }

def extract_metrics(line, kind):
    if sys.argv[2] == 'v1':
        # Train loss: 0.500, macro f1: 0.741, micro f1: 0.777, accuracy: 0.777, precision: 0.743, recall: 0.568
        metrics = re.search(kind + r' loss: (\d+.\d+), macro f1: (\d+.\d+), micro f1: (\d+.\d+), accuracy: (\d+.\d+), precision: (\d+.\d+), recall: (\d+.\d+)', line)
    elif sys.argv[2] == 'v2':
        # Test Unseen loss: 0.786, macro f1: 0.594, micro f1: 0.616, accuracy: 0.616, precision: 0.488, recall: 0.516, roc_auc: 0.596
        metrics = re.search(kind + r' loss: (\d+.\d+), macro f1: (\d+.\d+), micro f1: (\d+.\d+), accuracy: (\d+.\d+), precision: (\d+.\d+), recall: (\d+.\d+), roc_auc: (\d+.\d+)', line)
    if metrics is not None:
        return {
            "loss": float(metrics.group(1)),
            "macro_f1": float(metrics.group(2)),
            "micro_f1": float(metrics.group(3)),
            "accuracy": float(metrics.group(4)),
            "precision": float(metrics.group(5)),
            "recall": float(metrics.group(6))
        } if sys.argv[2] == 'v1' else {
            "loss": float(metrics.group(1)),
            "macro_f1": float(metrics.group(2)),
            "micro_f1": float(metrics.group(3)),
            "accuracy": float(metrics.group(4)),
            "precision": float(metrics.group(5)),
            "recall": float(metrics.group(6)),
            "roc_auc": float(metrics.group(7))
        }
    else:
        metrics = re.search(kind + r' loss: (\d+.\d+)', line)
        if metrics is not None:
            return {
                "loss": float(metrics.group(1)),
            }
    return None

def add_metrics(metrics, new_metrics):
    for key in new_metrics.keys():
        metrics[key].append(new_metrics[key])
    return metrics


def plot_metrics(train_metrics, eval_metrics, eval_seen_metrics, eval_unseen_metrics, test_metric, test_seen_metric, test_unseen_metric):
    if sys.argv[2] == 'v1':
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    elif sys.argv[2] == 'v2':
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, key in enumerate(train_metrics.keys()):
        axs[i // 3, i % 3].plot(train_metrics[key], c='blue', label='Train')
        axs[i // 3, i % 3].plot(eval_metrics[key], c='orange', label='Evaluation')
        axs[i // 3, i % 3].plot(eval_seen_metrics[key], c='purple', label='Evaluation Seen')
        axs[i // 3, i % 3].plot(eval_unseen_metrics[key], c='pink', label='Evaluation Unseen')
        if test_metric is not None:
            axs[i // 3, i % 3].axhline(y=test_metric[key], c='red', label='Test')
        if test_seen_metric is not None:
            axs[i // 3, i % 3].axhline(y=test_seen_metric[key], c='green', label='Test Seen')
        if test_unseen_metric is not None:
            axs[i // 3, i % 3].axhline(y=test_unseen_metric[key], c='brown', label='Test Unseen')
        axs[i // 3, i % 3].set_title(key)
        axs[i // 3, i % 3].legend()
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), 'metrics.png'))

train_metrics = get_empty_metrics()
eval_metrics = get_empty_metrics()
eval_seen_metrics = get_empty_metrics()
eval_unseen_metrics = get_empty_metrics()
test_metrics = None
test_seen_metrics = None
test_unseen_metrics = None

with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        train_metric = extract_metrics(line, 'Train')
        eval_metric = extract_metrics(line, 'Evaluation')
        eval_seen_metric = extract_metrics(line, 'Validation Seen')
        eval_unseen_metric = extract_metrics(line, 'Validation Unseen')
        test_metric = extract_metrics(line, 'Test')
        test_seen_metric = extract_metrics(line, 'Test Seen')
        test_unseen_metric = extract_metrics(line, 'Test Unseen')
        if train_metric is not None:
            train_metrics = add_metrics(train_metrics, train_metric)
        if eval_metric is not None:
            eval_metrics = add_metrics(eval_metrics, eval_metric)
        if eval_seen_metric is not None:
            eval_seen_metrics = add_metrics(eval_seen_metrics, eval_seen_metric)
        if eval_unseen_metric is not None:
            eval_unseen_metrics = add_metrics(eval_unseen_metrics, eval_unseen_metric)
        if test_metric is not None:
            test_metrics = test_metric
        if test_seen_metric is not None:
            test_seen_metrics = test_seen_metric
        if test_unseen_metric is not None:
            test_unseen_metrics = test_unseen_metric

plot_metrics(train_metrics, eval_metrics, eval_seen_metrics, eval_unseen_metrics, test_metrics, test_seen_metrics, test_unseen_metrics)
