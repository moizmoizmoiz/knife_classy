import warnings
import pandas as pd
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import argparse
import sys
import time
import threading
from utils import *
from data import knifeDataset

warnings.filterwarnings('ignore')

# Helper Functions
def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return precision, recall, f1

def topk_labels_accuracy(preds, labels, k=5, n_classes=192):
    topk_acc = np.zeros(n_classes)
    count = np.zeros(n_classes)

    topk_preds = preds.topk(k, dim=1)[1]
    for i in range(n_classes):
        label_mask = (labels == i)
        if label_mask.any():
            selected_preds = topk_preds[label_mask]
            topk_acc[i] += ((selected_preds == i).sum(dim=1).float().mean().item())
            count[i] += 1

    topk_acc /= np.maximum(count, 1)  # Avoid division by zero
    top_k_accuracy = [(i, acc) for i, acc in enumerate(topk_acc) if count[i] > 0]
    top_k_accuracy.sort(key=lambda x: x[1], reverse=True)
    return top_k_accuracy[:k]

def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

# Evaluate Ensemble Function
def evaluate_ensemble(val_loader, model1, model2, n_classes=192):
    model1.eval()
    model2.eval()
    all_preds = []
    all_labels = []
    top_k_accuracies = []
    map = AverageMeter()

    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            # Get predictions from both models
            preds1 = model1(img).softmax(1)
            preds2 = model2(img).softmax(1)

            # Average predictions
            averaged_preds = (preds1 + preds2) / 2

            all_preds.append(averaged_preds.argmax(dim=1).cpu().numpy())
            all_labels.append(label.cpu().numpy())

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(averaged_preds, label)
            map.update(valid_map5, img.size(0))

            top_k_accuracies.extend(topk_labels_accuracy(averaged_preds, label, n_classes=n_classes))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1 = compute_metrics(all_preds, all_labels)
    top_k_accuracy = sorted(top_k_accuracies, key=lambda x: x[1], reverse=True)[:5]

    return map.avg, precision, recall, f1, top_k_accuracy

# Loading Animation
def loading_animation(flag):
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    while not flag.is_set():
        for char in spinner:
            sys.stdout.write('\r' + char)
            sys.stdout.flush()
            time.sleep(0.7)
    sys.stdout.write('\r     \r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1_path', type=str, required=True)
    parser.add_argument('--model2_path', type=str, required=True)

    args = parser.parse_args()

    model1_path = args.model1_path
    model2_path = args.model2_path

    # Load models
    model1 = timm.create_model('tf_efficientnet_b7', pretrained=False, num_classes=192)
    model1.load_state_dict(torch.load(model1_path))
    model1.cuda()

    model2 = timm.create_model('resnet34', pretrained=False, num_classes=192)
    model2.load_state_dict(torch.load(model2_path))
    model2.cuda()

    # Load test data
    test_files = pd.read_csv("test.csv")
    test_gen = knifeDataset(test_files, mode="val")
    test_loader = DataLoader(test_gen, batch_size=32, shuffle=False, pin_memory=True, num_workers=8)

    flag = threading.Event()

    # Evaluate Ensemble Model
    def evaluate_ensemble_model(test_loader, model1, model2, flag):
        map, precision, recall, f1, top_k_accuracy = evaluate_ensemble(test_loader, model1, model2, n_classes=192)
        print("\nmAP = ", map)
        print("\nWeighted Precision: ", precision)
        print("\nWeighted Recall: ", recall)
        print("\nWeighted F1 Score: ", f1)
        print("\nTop-K Accuracy Performing Labels: ", top_k_accuracy)
        flag.set()

    loading_thread = threading.Thread(target=loading_animation, args=(flag,))
    loading_thread.start()

    evaluate_thread = threading.Thread(target=evaluate_ensemble_model, args=(test_loader, model1, model2, flag))
    evaluate_thread.start()

    evaluate_thread.join()
    loading_thread.join()
