## import libraries for training
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import torch.nn as nn
import torch.nn.functional as F

from utils import *

warnings.filterwarnings('ignore')

# Validating the model
from torchvision.utils import save_image
import os
import argparse

#####
import timm
import torch.nn as nn


class OwnModel(nn.Module):
    def __init__(self, config):
        super(OwnModel, self).__init__()
        self.our_model_1 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True,
                                             num_classes=config.n_classes)
        self.our_model_2 = timm.create_model('resnet50', pretrained=True, num_classes=config.n_classes)

    def forward(self, x):
        # Forward pass through both models
        output1 = self.our_model_1(x)
        output2 = self.our_model_2(x)

        output = (output1 + output2) / 2
        return output


##

class MyOwnModelRedesigned(nn.Module):
    def __init__(self, config):
        super(MyOwnModelRedesigned, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fifth convolutional block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, config.n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.leaky_relu(self.fc4(x))
        x = self.fc3(x)
        return x

class MyOwnModel(nn.Module):
    def __init__(self, config):
        super(MyOwnModel, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fifth convolutional block
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 3 * 3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, config.n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


##

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_training', type=str, default='tf_efficientnet_b0')

    args = parser.parse_args()

    #model_pre_train = args.model_pre_train
    model_training = args.model_training

# def evaluate(val_loader, model, top_images_count=20, save_dir='./'):
#     model.cuda()
#     model.eval()
#     model.training = False
#     map = AverageMeter()
#     top_images_info = []  # This will store tuples of (probability, filename)
#     all_labels = []
#     all_preds = []
#
#     with torch.no_grad():
#         for batch_idx, (images, target, fnames) in enumerate(val_loader):
#             img = images.cuda(non_blocking=True)
#             label = target.cuda(non_blocking=True)
#             all_labels.append(label.cpu().numpy())
#
#             with torch.cuda.amp.autocast():
#                 logits = model(img)
#                 preds = logits.softmax(1)
#                 all_preds.append(preds.cpu().numpy())
#
#             valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
#             map.update(valid_map5, img.size(0))
#
#             # Collect the top probability for each image in the batch and its filename
#             top_probs, _ = preds.topk(1, dim=1, largest=True, sorted=True)
#             for idx, prob in enumerate(top_probs):
#                 # Get the filename for the image
#                 fname = fnames[idx]
#                 # Store the probability along with the filename
#                 top_images_info.append((prob.item(), fname))
#
#     # Now we sort the collected images by probability
#     top_images_info.sort(key=lambda x: x[0], reverse=True)
#     top_images_info = top_images_info[:top_images_count]
#
#     # Save the filenames of the top images to a CSV file
#     top_image_filenames = [fname for _, fname in top_images_info]
#     df = pd.DataFrame(top_image_filenames, columns=['filename'])
#     csv_path = os.path.join(save_dir, 'top_images.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"CSV file saved to {csv_path}")
#
#     return map.avg

## Edit here
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import os
import torch


def evaluate(val_loader, model, top_images_count=20, save_dir='/content/drive/MyDrive/EEEM066/logs/'):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    top_images_info = []  # This will store tuples of (probability, filename)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            all_labels.append(label.cpu().numpy())

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
                all_preds.append(preds.cpu().numpy())

            # Assuming map_accuracy is defined elsewhere and updates the AverageMeter
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))

            # Collect the top probability for each image in the batch and its filename
            top_probs, _ = preds.topk(1, dim=1, largest=True, sorted=True)
            for idx, prob in enumerate(top_probs):
                # Get the filename for the image
                fname = fnames[idx]
                # Store the probability along with the filename
                top_images_info.append((prob.item(), fname))

    # Flatten all labels and predictions
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Convert predictions to class labels
    all_pred_labels = np.argmax(all_preds, axis=1)

    # Calculate precision, recall, and F1 score
    precision, recall, f1score, _ = precision_recall_fscore_support(
        all_labels, all_pred_labels, average='macro'
    )

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1score:.4f}')

    # Sort the collected images by probability and save to CSV
    top_images_info.sort(key=lambda x: x[0], reverse=True)
    top_images_info = top_images_info[:top_images_count]

    top_image_filenames = [fname for _, fname in top_images_info]
    df = pd.DataFrame(top_image_filenames, columns=['filename'])
    csv_path = os.path.join(save_dir, 'top_images.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to {csv_path}")

    return map.avg


## Computing the mean average precision, accuracy
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5


######################## load file and get splits #############################
print('reading test file')
test_files = pd.read_csv("test.csv")
print('Creating test dataloader')
test_gen = knifeDataset(test_files, mode="val")
test_loader = DataLoader(test_gen, batch_size=32, shuffle=False, pin_memory=True, num_workers=8)

print('loading trained model')


model = timm.create_model(model_training, pretrained=True, num_classes=config.n_classes)
model.load_state_dict(torch.load('/content/drive/MyDrive/EEEM066/logs/tf_efficientnet_b030.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Training #################################
print('Evaluating trained model')
map = evaluate(test_loader, model)
print("mAP =", map)