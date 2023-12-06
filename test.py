## import libraries for training
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
from utils import *
import argparse
import sys
import time
import threading
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_training', type=str, default='mobilevit_xxs')
    parser.add_argument('--chec    kpoint
    ', type=str, default='
    10
    ')

    args = parser.parse_args()

    model_training = args.model_training
    checkpoint = args.checkpoint


# Validating the model
def evaluate(val_loader, model):
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


def loading_animation(flag):
    spinner = ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘']
    while not flag.is_set():
        for char in spinner:
            sys.stdout.write('\r' + char)  # Carriage return before character
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write('\r     \r')

######################## load file and get splits #############################
print('Reading test file..')
test_files = pd.read_csv("test.csv")
print('Creating test dataloader')
test_gen = knifeDataset(test_files,mode="val")
test_loader = DataLoader(test_gen,batch_size=32, shuffle=False, pin_memory=True, num_workers=8)

print('loading trained model')
model = timm.create_model(model_training, pretrained=True,num_classes=config.n_classes)
model.load_state_dict(torch.load("/content/drive/MyDrive/EEEM066/logs/"+model_training+checkpoint+".pt"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
flag = threading.Event()


def evaluate_model(test_loader, model, flag):
    # Replace with your evaluate function logic
    map = evaluate(test_loader, model)
    print("\nmAP =", map)
    flag.set()



loading_thread = threading.Thread(target=loading_animation, args=(flag,))
loading_thread.start()

# Start the model evaluation
evaluate_thread = threading.Thread(target=evaluate_model, args=(test_loader, model, flag))
evaluate_thread.start()

evaluate_thread.join()  # Wait for the evaluation to complete
loading_thread.join()  # Ensure the loading animation stops after the evaluation

# print("Evaluation complete.")
# ############################# Training #################################
# print('Evaluating '+model_training+' at checkpoint number: '+checkpoint)
# map = evaluate(test_loader,model)
# print("mAP =",map)
    
   
