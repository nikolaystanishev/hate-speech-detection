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
        "recall": []
    }

def extract_metrics(line, kind):
    # Train loss: 0.500, macro f1: 0.741, micro f1: 0.777, accuracy: 0.777, precision: 0.743, recall: 0.568
    metrics = re.search(kind + r' loss: (\d+.\d+), macro f1: (\d+.\d+), micro f1: (\d+.\d+), accuracy: (\d+.\d+), precision: (\d+.\d+), recall: (\d+.\d+)', line)
    if metrics is not None:
        return {
            "loss": float(metrics.group(1)),
            "macro_f1": float(metrics.group(2)),
            "micro_f1": float(metrics.group(3)),
            "accuracy": float(metrics.group(4)),
            "precision": float(metrics.group(5)),
            "recall": float(metrics.group(6))
        }
    return None

def add_metrics(metrics, new_metrics):
    for key in metrics.keys():
        metrics[key].append(new_metrics[key])
    return metrics


def plot_metrics(train_metrics, eval_metrics):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i, key in enumerate(train_metrics.keys()):
        axs[i // 3, i % 3].plot(train_metrics[key], c='blue', label='Train')
        axs[i // 3, i % 3].plot(eval_metrics[key], c='orange', label='Evaluation')
        axs[i // 3, i % 3].set_title(key)
        axs[i // 3, i % 3].legend()
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), 'metrics.png'))

train_metrics = get_empty_metrics()
eval_metrics = get_empty_metrics()


with open(sys.argv[1], 'r') as f:
    for line in f.readlines():
        train_metric = extract_metrics(line, 'Train')
        eval_metric = extract_metrics(line, 'Evaluation')
        if train_metric is not None:
            train_metrics = add_metrics(train_metrics, train_metric)
        if eval_metric is not None:
            eval_metrics = add_metrics(eval_metrics, eval_metric)


plot_metrics(train_metrics, eval_metrics)
