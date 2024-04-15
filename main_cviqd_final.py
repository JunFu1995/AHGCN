import os
import argparse

# Training settings
parser = argparse.ArgumentParser(description='VR Image Quality Assessment')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--use_hgcn', type=int, default=1)
parser.add_argument('--use_ms', type=int, default=1)
parser.add_argument('--use_fc', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=41)
parser.add_argument('--total_iterations', type=int, default=1000000)
parser.add_argument('--batch_size', '-b', type=int, default=16, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-3, metavar=' LR', help='learning rate (default: 0.001)')
parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=4)
parser.add_argument('--number_gpus', '-ng', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--save', '-s', default='./save', type=str, help='directory for saving')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--skip_training', default=False, action='store_true')
parser.add_argument('--skip_validation', action='store_true')
parser.add_argument('--resume', default='./live2depoch_0058.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help="Log every n batches")

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)

# 157
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "%d" % args.gpu
import time

import torch
import math
import numpy as np
import cv2
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable
from torchvision import models
import scipy.io as scio
from scipy import stats
# from scipy.misc import imsave
import torch.nn as nn
import random

import utils
from datasets.cviqd import get_dataset
from model.cviqd import OIQANet
import numpy as np 


import warnings
warnings.filterwarnings("ignore")

seed = 10 
print("Random Seed: ", seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Remove randomness (may be slower on Tesla GPUs) 
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if args.inference:
    args.skip_validation = True
    args.skip_training = True
    args.total_epochs = 1
    args.inference_dir = "{}/inference".format(args.save)


kwargs = {'num_workers': args.number_workers}
if not args.skip_training:
    train_set = get_dataset(is_training=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = get_dataset(is_training=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **kwargs)

res_net = models.resnet18(pretrained=False)
model = OIQANet(res_net, args.k, args.use_hgcn,args.use_ms,args.use_fc).cuda()


resnet_params = list(map(id, model.resnet.parameters()))
base_params = filter(lambda p: id(p) not in resnet_params, model.parameters())
optimizer = optim.Adam([
    {'params': base_params},
    {'params': model.resnet.parameters(), 'lr': 1e-6}], lr=args.lr, weight_decay=0.25) 
scheduler = LS.MultiStepLR(optimizer, milestones=[10, 40, 80, 120, 200, 300], gamma=0.4) 
scheduler.last_epoch = args.start_epoch


def train(model,epoch, iteration):
    model=model.train()
    scheduler.step()
    end = time.time()
    log = [0 for _ in range(1)]
    for batch_idx, batch in enumerate(train_loader):
        data, label, _, A = batch
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        A = Variable(A.cuda())
        optimizer.zero_grad()
        _, _, batch_info = model(data, label, A, requires_loss=True)
        batch_info.backward()
        optimizer.step()
        #break 
        log = [log[i] + batch_info.item() * len(data) for i in range(1)]
        iteration += 1
    
    log = [log[i] / len(train_loader.dataset) for i in range(1)]
    epoch_time = time.time() - end
    end = time.time()
    print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, log[0]))
    print('LogTime: {:.4f}s'.format(epoch_time))
    return log


def eval(model):
    model=model.eval()
    log = 0
    score_list=[]
    label_list=[]
    name_list=[]
    for batch_idx, batch in enumerate(test_loader):
        data, label, imgname, A = batch
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        A = Variable(A.cuda())

        score, label = model(data, label, A, requires_loss=False)

        score = score.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        score = np.mean(score)
        label = np.mean(label)
        res = (score - label)*(score - label)
        score_list.append(score)
        label_list.append(label)
        name_list.append(imgname[0])

        ## release memory
        torch.cuda.empty_cache()

        log += res

    log = log / len(test_loader)
    print('Average LOSS: %.2f' % (log))
    score_list = np.reshape(np.asarray(score_list), (-1,))
    label_list = np.reshape(np.asarray(label_list), (-1,))
    name_list = np.reshape(np.asarray(name_list), (-1,))
    mat = {'score': score_list, 'label': label_list, 'name': name_list}

    srocc = stats.spearmanr(label_list, score_list)[0]
    krocc = stats.stats.kendalltau(label_list, score_list)[0]
    plcc = stats.pearsonr(label_list, score_list)[0]
    rmse = np.sqrt(((label_list - score_list) ** 2).mean())
    print('SROCC: %.4f, PLCC: %.4f, KROCC: %.4f, RMSE: %.4f\n' % (srocc, plcc, krocc, rmse))
    return srocc, plcc, rmse, mat

bb = 0 
if not args.skip_training:
    if args.resume:
        utils.load_model(model, args.resume)
        print('Train Load pre-trained model!')
    best = 0
    for epoch in range(args.start_epoch, args.total_epochs+1):
        iteration = (epoch-1) * len(train_loader) + 1
        log = train(model,epoch, iteration)
        #break 
        log2 = eval(model)

        srocc = log2[0]
        plcc = log2[1]
        current_cc = srocc + plcc
        checkpoint = os.path.join(args.save, 'cviqd_use_hgcn_%d_use_ms_%d_k_%d_use_fc_%d'%(args.use_hgcn, args.use_ms, args.k, args.use_fc))
        #utils.save_model(model, checkpoint, epoch, is_epoch=True)
        if current_cc > best:
            best = current_cc
            bb = (srocc, plcc) 
            utils.save_model(model, checkpoint, epoch, is_epoch=False)
            scio.savemat('./mat/cviqd_use_hgcn_%d_use_ms_%d_k_%d_use_fc_%d.mat'%(args.use_hgcn, args.use_ms, args.k, args.use_fc), log2[-1])
    print(bb)
else:
    print('Test Load pre-trained model!')
    utils.load_model(model, args.resume)
    eval()

