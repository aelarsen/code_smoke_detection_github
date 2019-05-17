import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import random
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

import pickle
from torchvision.utils import make_grid
import math
from PIL import Image, ImageOps, ImageEnhance
import numbers
import torchnet
import torchnet.meter as meter

# last modified 04/05/2019

#######################################################################
# calc_acc():
#
#  Inputs: predicted values and targets for a time slice
#  Outputs: correct pixels / total pixels
#
########################################################################

def calc_acc(p, t):
    correct_pixels = (p == t).sum().to(dtype=torch.float)
    total_pixels = (t == t).sum().to(dtype=torch.float)
    return correct_pixels / total_pixels


#######################################################################
# calc_iou():
#
#  Inputs: predicted values and targets for a time slice
#  Outputs: iou_tp, iou_tn for the time slice
#
########################################################################

def calc_iou(p, t):
    current = confusion_matrix(t.numpy().flatten(), p.numpy().flatten(), labels=[0, 1])
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)  # rows
    predicted_set = current.sum(axis=0)  # columns
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    tp = IoU[1]
    tn = IoU[0]
    if math.isnan(tp):
        tp = 0
    if math.isnan(tn):
        tn = 0
    iou = (tp + tn) / 2
    return tp, tn, iou

#######################################################################
# calc_wiou():
#
#  Inputs: predicted values and targets for each time slice in a batch
#  Outputs: iou, iou_tp (wiou_tp), iou_tn (wiou_tn) for each time slice
#
########################################################################


def calc_wiou(p, t):
    N = len(t)
    w_TP = np.zeros(N)
    w_TN = np.zeros(N)
    IoU = np.zeros(N)
    IoU_TP = np.zeros(N)
    IoU_TN = np.zeros(N)
    for j in range(N):
        pred, tar = p[j], t[j]
        w_TP[j] = sum(sum(tar))
        w_TN[j] = sum(sum(1 - tar))
        IoU_TP[j], IoU_TN[j], IoU[j] = calc_iou(p=pred, t=tar)
    wIoU_TP = sum(w_TP * IoU_TP) / sum(w_TP)
    if math.isnan(wIoU_TP):
        wIoU_TP = 0
    wIoU_TN = sum(w_TN * IoU_TN) / sum(w_TN)
    if math.isnan(wIoU_TN):
        wIoU_TN = 0
    return wIoU_TP, wIoU_TN, IoU_TP, IoU_TN, IoU


#######################################################################
# round_metrics():
#
#  Inputs: all performance metrics, rounding size
#  Outputs: rounded and converted to percents
########################################################################

def round_metrics(a, b, c, d, e, f, r):
    return round(a, r)*100, round(b, r)*100, round(c, r)*100, round(d, r)*100, round(e, r)*100, round(f, r)*100


#######################################################################
#  Make plots with performance metrics
#
########################################################################

# load results
results = pickle.load(open("/Volumes/DATA/Project-3-Data/NT/results_smoothed_666.p", "rb"))

# load test data
test_dataset = pickle.load(open("/Volumes/DATA/Project-3-Data/NT/test_dataset_ready.p", "rb"))
train_dataset = pickle.load(open("/Volumes/DATA/Project-3-Data/NT/train_dataset_ready.p", "rb"))
batch_size = 5
random.seed(1319)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
random.seed(1234)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# save data in the test set
test_dat = []
for dat, _ in test_loader:
    test_dat.append(dat)

# calculate metrics and make plots
for i in range(len(results[0])):
    X = test_dat[i]        # data (batch_size, 7, 161, 105)
    Y = results[0][i]      # targets (batch_size, 161, 105)
    YHat = results[3][i]   # predictions (batch_size, 161, 105) TODO replace W's
    Z = results[1][i]      # out_convFinal (batch_size, 161, 105)
    wiou_tp, wiou_tn, iou_tp, iou_tn, iou = calc_wiou(p=YHat, t=Y)
    for j in range(len(X)):
        B1 = X[j, 0, :, :].detach().numpy()
        B2 = X[j, 1, :, :].detach().numpy()
        B3 = X[j, 2, :, :].detach().numpy()
        B4 = X[j, 3, :, :].detach().numpy()
        B5 = X[j, 4, :, :].detach().numpy()
        temp = X[j, 5, :, :].detach().numpy()
        frp = X[j, 6, :, :].detach().numpy()
        y = Y[j, :, :].detach().numpy()
        z = Z[j, :, :].detach().numpy()
        yhat = YHat[j, :, :].detach().numpy()  # TODO replace W's
        acc = calc_acc(torch.from_numpy(yhat).long(), torch.from_numpy(y).long())
        metrics = round_metrics(a=acc.detach().numpy(), b=iou[j], c=iou_tp[j], d=wiou_tp, e=iou_tn[j], f=wiou_tn, r=3)
        plt.ion()
        plt.subplot(2, 5, 1)
        plt.suptitle('Acc={}, IoU={}, IoU_TP={}({}), IoU_TN={}({})'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))
        plt.imshow(y)
        plt.title("Ground Truth")
        plt.axis('off')
        plt.subplot(2, 5, 2)
        plt.imshow(yhat)
        plt.title("Prediction")
        plt.axis('off')
        plt.subplot(2, 5, 3)
        plt.imshow(z)
        plt.title("Final Conv")
        plt.axis('off')
        plt.subplot(2, 5, 4)
        plt.imshow(B1)
        plt.title("Red")
        plt.axis('off')
        plt.subplot(2, 5, 5)
        plt.imshow(B2)
        plt.title("Green")
        plt.axis('off')
        plt.subplot(2, 5, 6)
        plt.imshow(B3)
        plt.title("Blue")
        plt.axis('off')
        plt.subplot(2, 5, 7)
        plt.imshow(B4)
        plt.title("Infrared1")
        plt.axis('off')
        plt.subplot(2, 5, 8)
        plt.imshow(B5)
        plt.title("Infrared2")
        plt.axis('off')
        plt.subplot(2, 5, 9)
        plt.imshow(temp)
        plt.title("Temperature")
        plt.axis('off')
        plt.subplot(2, 5, 10)
        plt.imshow(frp)
        plt.title("Hotspots")
        plt.axis('off')
        plt.show()
        plt.savefig('all_plots_smoothed_666/fig{}{}.png'.format(i, j))
        plt.close("all")

#######################################################################
#  Compare results to the Logistic Regression
#
########################################################################

# load LR results
results_lr = pd.read_csv("/Volumes/DATA/Project-3-Data/NT/roc_data_04052019.csv")

# unpack FCN targets and predictions
all_targets = results[0]  # 0's and 1's
all_output = results[2]   # sigmoid

# reshape the first 64 batches of the output (5/batch) to one object with dim (5*64)x161x105, column-wise
y_pred1 = torch.stack(all_output[0:64]).detach().numpy().reshape((320, 161, 105), order='F')

# detach last batch (4/batch)
y_pred2 = all_output[64].detach().numpy()

# concatenate both to get a single object with dim (4+5*64)x161x105, column-wise
y_pred = np.concatenate((y_pred1, y_pred2), axis=0).reshape((324, 161*105), order='F')
print(y_pred.shape)

# repeat for the targets
y_true1 = torch.stack(all_targets[0:64]).detach().numpy().reshape((320, 161, 105), order='F')
y_true2 = all_targets[64].detach().numpy()
y_true = np.concatenate((y_true1, y_true2), axis=0).reshape((324, 161*105), order='F')
print(y_true.shape)

# compute AUC
y_pred = y_pred.reshape([-1])
y_true = y_true.reshape([-1])
auc_fcn = roc_auc_score(y_true, y_pred)
auc_fcn

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# compare to logistic regression
y_true_lr = results_lr.iloc[:, 2].values
y_pred_lr = results_lr.iloc[:, 1].values
fpr1, tpr1, thresholds1 = roc_curve(y_true_lr, y_pred_lr)
auc_lr = roc_auc_score(y_true_lr, y_pred_lr)
auc_lr

plt.ion()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='FCN (area = %0.2f)' % auc_fcn)
plt.plot(fpr1, tpr1, color='lightblue',
         lw=lw, label='LR (area = %0.2f)' % auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc_curve_04052019.png')

#######################################################################
#  Compute performance metrics
#
########################################################################

# initialize variables
aa = 0
bb = 0
cc = 0
dd = 0
ee = 0
ff = 0

# number of images in the test set
N = 324

# loop through each batch and calculate performance metrics
for i in range(len(results[0])):
    X = test_dat[i]        # data (batch_size, 7, 161, 105)
    Y = results[0][i]      # targets (batch_size, 161, 105)
    YHat = results[3][i]   # predictions (batch_size, 161, 105)
    Z = results[1][i]      # out_convFinal (batch_size, 161, 105)
    wiou_tp, wiou_tn, iou_tp, iou_tn, iou = calc_wiou(p=YHat, t=Y)
    for j in range(len(X)):
        y = Y[j, :, :].detach().numpy()
        yhat = YHat[j, :, :].detach().numpy()
        acc = calc_acc(torch.from_numpy(yhat).long(), torch.from_numpy(y).long())
        metrics = round_metrics(a=acc.detach().numpy(), b=iou[j], c=iou_tp[j], d=wiou_tp, e=iou_tn[j], f=wiou_tn, r=3)
        aa += metrics[0]/N
        bb += metrics[1]/N
        cc += metrics[2]/N
        dd += metrics[3]/N
        ee += metrics[4]/N
        ff += metrics[5]/N
