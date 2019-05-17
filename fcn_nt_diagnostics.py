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

'''

    - builds on fcn_nt_vary_train_size
    - corresponds to code from fcn_4_all.py
    
    
'''


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class SmokePlumeDataset(Dataset):
    """
        Input
            csv_file: .csv file
            ranges: beginning and end indices of each variable (y, b1, b2, b3, b4, b5, temp, frp)
        Output
            tensor
    """

    def __init__(self, csv_file, ranges,
                 transform=transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(
                                                   mean=(0.15806109, 0.14479933, 0.16630965, 0.26673442, 0.34228491,
                                                         309.71709941, 0.04669192),
                                                   std=(0.05717147, 0.05894414, 0.06644581, 0.08281785, 0.09312279,
                                                        10.72931894, 1.30413502))])):
        df = pd.read_csv(csv_file)
        self.transform = transform
        self.X = np.empty((df.shape[0], 161, 105, 7))
        for i in range(len(ranges)):
            if i == 0:
                self.Y = df.iloc[:, ranges[i][0]:ranges[i][1]].values.reshape((-1, 161, 105), order='F')[:, :, :, None]
            else:
                self.X[:, :, :, i - 1] = df.iloc[:, ranges[i][0]:ranges[i][1]].values.reshape((-1, 161, 105), order='F')

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.transform(self.X[idx]), self.Y[idx]
        else:
            return self.transform(self.X[idx])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(33280, 2)

    def forward(self, x):
        in_size = x.size(0)
        x1 = F.relu(self.conv1(x))
        x1 = self.mp(x1)  # size=(N, 32, x.H/2, x.W/2)
        x2 = F.relu(self.conv2(x1))
        x2 = self.mp(x2)  # size=(N, 64, x.H/4, x.H/4)
        x3 = F.relu(self.conv3(x2))
        x3 = self.mp(x3)  # size=(N, 128, x.H/8, x.H/8)
        x4 = x3.view(in_size, -1)
        x4 = self.fc(x4)  # size=(N, n_class)
        y = F.log_softmax(x4)  # size=(N, n_class)
        return x1, x2, x3, x4, y


class FCN8s(nn.Module):
    # Po-Chih Huang
    def __init__(self, pretrained_net, n_class):
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.classifier = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x3 = output[2]
        x2 = output[1]
        x1 = output[0]
        score = self.relu(self.deconv1(x3))
        score = self.bn1(score + x2)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x1)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.classifier(score)
        return score


def train(epoch):
    model.train()
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).float(), Variable(target).float()
        optimizer.zero_grad()
        output = model(data)
        output2 = nn.functional.sigmoid(output)
        loss = loss_fn(output2, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.data[0])
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    return np.array(total_loss).mean()

def iou(p, t):
    current = confusion_matrix(t.numpy().flatten(), p.numpy().flatten(), labels=[0, 1])
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)  # rows
    predicted_set = current.sum(axis=0)  # columns
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU), current, IoU[0], IoU[1]


def pixel_acc(p, t):
    correct_pixels = (p == t).sum().to(dtype=torch.float)
    total_pixels = (t == t).sum().to(dtype=torch.float)
    return correct_pixels / total_pixels


def test(threshold):
    model.eval()
    n_batches = 0
    total_acc = []
    iou_mn = []
    iou_tp = []
    iou_tn = []
    #all_targets = []
    #all_out = []
    #all_output = []
    #all_pred = []
    for data, target in test_loader:
        n_batches += 1
        data, target = \
            Variable(data, volatile=True).float(), Variable(target, volatile=True).float()
        out = model(data)
        output = nn.functional.sigmoid(out)
        # b, _, h, w = output.size()  # batch, _, height, width
        # pred = output.permute(0, 2, 3, 1).contiguous().view(-1, n_class).max(1)[1].view(b, h, w)
        pred = (output[:, 1, :, :] > threshold).float() * 1
        total_acc.append(pixel_acc(p=pred.long(), t=target[:, 1, :, :].long()))
        iou_output = iou(p=pred.long(), t=target[:, 1, :, :].long())  # add iou_w() that takes pred and target
        iou_mn.append(iou_output[0])
        iou_tn.append(iou_output[2])
        iou_tp.append(iou_output[3])
        # if doSave:
        #     all_targets.append(target[:, 1, :, :])
        #     all_out.append(out[:, 1, :, :])
        #     all_output.append(output[:, 1, :, :])
        #     all_pred.append(pred)
    print('\nTest Epoch: {}\t Mean Batch Accuracy: {}%, Mean Batch IoU: {}%, TP IoU: {}%, TN IoU = {}%\n'.format(
        epoch, round(100 * np.array(total_acc).mean(), 2), round(100 * np.array(iou_mn).mean(), 2),
        round(100 * np.array(iou_tp).mean(), 2), round(100 * np.array(iou_tn).mean(), 2)))
    return np.array(total_acc).mean(), np.array(iou_mn).mean(), np.array(iou_tp).mean(), np.array(iou_tn).mean()


def calc_acc(p, t):
    correct_pixels = (p == t).sum().to(dtype=torch.float)
    total_pixels = (t == t).sum().to(dtype=torch.float)
    return correct_pixels / total_pixels


def calc_iou(p, t):
    current = confusion_matrix(t.numpy().flatten(), p.numpy().flatten(), labels=[0, 1])
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)  # rows
    predicted_set = current.sum(axis=0)     # columns
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    tp = IoU[1]
    tn = IoU[0]
    iou = (tp + tn)/2
    if math.isnan(tp):
        tp = 0
    if math.isnan(tn):
        tn = 0
    return tp, tn, iou


def calc_wiou(p, t):
    N = len(t)
    w_TP = np.zeros(N)
    w_TN = np.zeros(N)
    IoU = np.zeros(N)
    IoU_TP = np.zeros(N)
    IoU_TN = np.zeros(N)
    for j in range(N):
        pred, tar = p[j], t[j]
        # w_TP[j] = 1 / sum(sum(tar))
        # w_TN[j] = 1 / sum(sum(1 - tar))
        w_TP[j] = sum(sum(tar))
        w_TN[j] = sum(sum(1 - tar))
        IoU_TP[j], IoU_TN[j], IoU[j] = calc_iou(p=pred, t=tar)
    wIoU_TP = sum(w_TP * IoU_TP) / sum(w_TP)
    wIoU_TN = sum(w_TN * IoU_TN) / sum(w_TN)
    return wIoU_TP, wIoU_TN, IoU_TP, IoU_TN, IoU


def round_metrics(a, b, c, d, e, f, r):
    return round(a, r)*100, round(b, r)*100, round(c, r)*100, round(d, r)*100, round(e, r)*100, round(f, r)*100


#######################################################################
#  Network settings
#
########################################################################

batch_size = 5
n_class = 2
n_epochs = 20
n_train_sets = 10

# set threshold
thres1 = torch.Tensor([.666])  # try: 0, -.2, -.1, .1, .2, .3, .4
thres2 = torch.Tensor([.5])

# metrics for each epoch
metrics_loss = np.zeros((n_train_sets, n_epochs))
metrics_acc = np.zeros((n_train_sets, n_epochs))
metrics_iou = np.zeros((n_train_sets, n_epochs))
metrics_i_tp = np.zeros((n_train_sets, n_epochs))
metrics_i_tn = np.zeros((n_train_sets, n_epochs))


#######################################################################
#  Load data and run model
#
########################################################################

for j in range(n_train_sets):
    print("--Train Set {}--".format(j+1))

    # get smoke plume data
    ranges = [[1 - 1, 16905], [16906 - 1, 33810], [33811 - 1, 50715], [50716 - 1, 67620], [67621 - 1, 84525],
              [84526 - 1, 101430], [101431 - 1, 118335], [118336 - 1, 135240]]
    train_dataset = SmokePlumeDataset("/Volumes/DATA/Project-3-Data/NT/Diagnostics/Set{}/train.csv".format(j+1),
                                      ranges=ranges)
    test_dataset = SmokePlumeDataset("/Volumes/DATA/Project-3-Data/NT/Diagnostics/Set{}/test.csv".format(j+1),
                                     ranges=ranges)

    # set up for average kernel smoothing
    W = torch.empty(1, 1, 3, 3)
    W[0, 0, :, :] = torch.Tensor([[.111, .111, .111], [.111, .111, .111], [.111, .111, .111]])  # .111~1/9
    W = torch.nn.Parameter(W)
    my_conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
    my_conv.weight = W

    # average kernel smoothing and one-hot encoding
    test_dataset_1 = list(test_dataset)
    for i in range(len(test_dataset_1)):
        example = test_dataset_1[i]
        dat = example[0]
        target = example[1]
        target0 = torch.FloatTensor(target)
        target1 = my_conv(torch.unsqueeze(target0.permute(2, 0, 1), 0))
        target2 = (target1 > thres1).float() * 1
        dif = target0[:, :, 0] - target2[0, 0, :, :]
        target22 = target2[0, :, :, :].permute(1, 2, 0)
        h, w, k = target22.shape
        target3 = torch.zeros(n_class, h, w)
        for c in range(n_class):
            target3[c][target22[:, :, 0] == c] = 1
        test_dataset_1[i] = dat, target3

    train_dataset_1 = list(train_dataset)
    for i in range(len(train_dataset_1)):
        example = train_dataset_1[i]
        dat = example[0]
        target = example[1]
        target0 = torch.FloatTensor(target)
        target1 = my_conv(torch.unsqueeze(target0.permute(2, 0, 1), 0))
        target2 = (target1 > thres1).float() * 1
        target22 = target2[0, :, :, :].permute(1, 2, 0)
        h, w, k = target22.shape
        target3 = torch.zeros(n_class, h, w)
        for c in range(n_class):
            target3[c][target22[:, :, 0] == c] = 1
        train_dataset_1[i] = dat, target3

    # data loader
    random.seed(1319)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_1, batch_size=batch_size, shuffle=True)
    random.seed(1234)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset_1, batch_size=batch_size, shuffle=False)

    # initialize model
    cnn_model = CNN()
    model = FCN8s(pretrained_net=cnn_model, n_class=n_class)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = nn.BCELoss()

    for epoch in range(n_epochs):
        results1 = train(epoch)
        results2 = test(threshold=thres2)
        metrics_loss[j, epoch] = results1
        metrics_acc[j, epoch] = results2[0]
        metrics_iou[j, epoch] = results2[1]
        metrics_i_tp[j, epoch] = results2[2]
        metrics_i_tn[j, epoch] = results2[3]


pickle.dump(metrics_loss, open('/Volumes/Data/Project-3-Data/NT/Diagnostics/loss.p', 'wb'))
pickle.dump(metrics_acc, open('/Volumes/Data/Project-3-Data/NT/Diagnostics/acc.p', 'wb'))
pickle.dump(metrics_iou, open('/Volumes/Data/Project-3-Data/NT/Diagnostics/iou.p', 'wb'))
pickle.dump(metrics_i_tp, open('/Volumes/Data/Project-3-Data/NT/Diagnostics/iou_tp.p', 'wb'))
pickle.dump(metrics_i_tn, open('/Volumes/Data/Project-3-Data/NT/Diagnostics/iou_tn.p', 'wb'))


loss = pickle.load(open("/Volumes/Data/Project-3-Data/NT/Diagnostics/loss.p", "rb"))
acc = pickle.load(open("/Volumes/Data/Project-3-Data/NT/Diagnostics/acc.p", "rb"))
i_mn = pickle.load(open('/Volumes/Data/Project-3-Data/NT/Diagnostics/iou.p', 'rb'))
i_tp = pickle.load(open('/Volumes/Data/Project-3-Data/NT/Diagnostics/iou_tp.p', 'rb'))
i_tn = pickle.load(open('/Volumes/Data/Project-3-Data/NT/Diagnostics/iou_tn.p', 'rb'))


df1 = pd.DataFrame(loss)
df1.to_csv("/Volumes/Data/Project-3-Data/NT/Diagnostics/loss.csv", header=None, index=False)

df2 = pd.DataFrame(acc)
df2.to_csv("/Volumes/Data/Project-3-Data/NT/Diagnostics/acc.csv", header=None, index=False)

df3 = pd.DataFrame(i_mn)
df3.to_csv("/Volumes/Data/Project-3-Data/NT/Diagnostics/i_mn.csv", header=None, index=False)

df4 = pd.DataFrame(i_tp)
df4.to_csv("/Volumes/Data/Project-3-Data/NT/Diagnostics/i_tp.csv", header=None, index=False)

df5 = pd.DataFrame(i_tn)
df5.to_csv("/Volumes/Data/Project-3-Data/NT/Diagnostics/i_tn.csv", header=None, index=False)

