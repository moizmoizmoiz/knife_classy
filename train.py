## import libraries for training
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
import numpy as np
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import *



warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='demo')
    parser.add_argument('--n_classes', type=int, default=192)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--model_training', type=str, default='mobilevit_xxs')
    parser.add_argument('--weight_decay', type=float, default=None)

    args = parser.parse_args()

    config.n_classes = args.n_classes
    config.img_weight = args.img_width
    config.img_height = args.img_height
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    model_training = args.model_training
    weight_decay = args.weight_decay
    run_name = args.run_name

    logdir = "/content/drive/MyDrive/EEEM066/logs/TensorBoard_Logs/"+model_training+run_name+datetime.now().strftime("%H%M")
    writer = SummaryWriter(logdir)

    #
    # current_datetime = datetime.datetime.now()
    # formatted_datetime = current_datetime.strftime("%Y-%m-%d--%H%M%S")
    # log_name = formatted_datetime+"_"+model_training+'-'+run_name+".txt"
    # sys.stdout = Logger(osp.join('/content/drive/MyDrive/EEEM066/logs/TensorBoard_Logs/', log_name))


## Writing the loss and results
# if not os.path.exists("/content/drive/MyDrive/EEEM066/logs/"):
#     os.mkdir("/content/drive/MyDrive/EEEM066/logs/")



## Training the model
def train(train_loader, model, criterion, optimizer, epoch, valid_accuracy, start):
    losses = AverageMeter()
    model.train()
    model.training = True
    for i, (images, target, fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(), images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        print('\r', end='', flush=True)
        message = '%s %5.1f %6.1f        │      %0.5f     │      %0.3f     │ %s' % (
            "train", i, epoch, losses.avg, valid_accuracy[0], time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)
    log.write("\n")
    log.write(message)
    writer.add_scalar('Loss/Epoch', losses.avg, epoch)
    writer.add_scalar('Val_Acc/Epoch', valid_accuracy[0], epoch)

    return [losses.avg]


# Validating the model
def evaluate(val_loader, model, criterion, epoch, train_loss, start):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)

            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))
            print('\r', end='', flush=True)
            message = '%s   %5.1f %6.1f        │      %0.5f     │      %0.3f     │ %s' % ( \
                "val", i, epoch, train_loss[0], map.avg, time_to_str((timer() - start), 'min'))
            print(message, end='', flush=True)
        log.write("\n")
        log.write(message)
    return [map.avg]


## Computing the mean average precision, accuracy
def map_accuracy(probs, truth, k=10):
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
train_imlist = pd.read_csv("/content/drive/MyDrive/EEEM066/knife_classy/train.csv")
train_gen = knifeDataset(train_imlist, mode="train")
train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
val_imlist = pd.read_csv("/content/drive/MyDrive/EEEM066/knife_classy/val.csv")
val_gen = knifeDataset(val_imlist, mode="val")
val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=8)

## Loading the model to run
model = timm.create_model(model_training, pretrained=True, num_classes=config.n_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Parameters #################################
if weight_decay:
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

else:
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,
                                               last_epoch=-1)


criterion = nn.CrossEntropyLoss().cuda()

############################# Training #################################
start_epoch = 0
val_metrics = [0]
scaler = torch.cuda.amp.GradScaler()
start = timer()
print(optimizer)
print(' ')
print(scheduler)
# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())


if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()

file_path = "/content/drive/MyDrive/EEEM066/logs/" + model_training + "_log_train.txt"
log.open(file_path, 'w')

# log.open("logs/%s_log_train.txt")

log.write('\n                  ' + model_training + '   params: ' + str(total_params) +'                \n\n')
log.write('Batch size: ' + str(config.batch_size) +
          '  Learning rate: ' + str(config.learning_rate) +
          '  Weight Decay: ' + str(weight_decay))

log.write("\n\n\n───────────────────── [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '─' * 21))
log.write('                           ┠───── Train ─────┼───── Valid ───┼─────────┨\n')
log.write('mode     iter     epoch    ┃       loss      │        mAP    │ time    ┃\n')
log.write('────────────────────────────────────────────────────────────────────────\n')


writer.add_text('Architecture', model_training)
writer.add_text('Parmas', str(total_params))
writer.add_text('LR', str(config.learning_rate))
writer.add_text('WeightDecay', str(weight_decay))
writer.add_text('Epochs', str(config.epochs))
writer.add_text('Batch Size', str(config.batch_size))

train_epoch_arr, train_loss_arr, train_map_avg_arr, train_i_arr = [], [], [], []
val_epoch_arr, val_loss_arr, val_map_avg_arr, val_i_arr = [], [], [], []


######## train
for epoch in range(0, config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)
    val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('Learning Rate', current_lr, epoch)
    print(f"  Current LR: {current_lr}")

    # Save the model after every 10 epochs
    if (epoch + 1) % 10 == 0:
        filename = f"/content/drive/MyDrive/EEEM066/logs/{model_training}{epoch + 1}.pt"
        torch.save(model.state_dict(), filename)


writer.close()