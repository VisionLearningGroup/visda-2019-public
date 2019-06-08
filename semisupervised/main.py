from __future__ import print_function

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34, resnet50
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='number of iterations to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME', choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization, S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before testing and saving a model')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real', metavar='B',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch', metavar='B',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi_all', choices=['multi_all'],
                    help='the name of dataset')

args = parser.parse_args()
print('dataset %s source %s target %s network %s' % (args.dataset, args.source, args.target, args.net))
source_loader, target_loader, target_loader_unl, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_lamda_%s' % (
                           args.method, args.net, args.source, args.target, args.lamda))
record_dir_train = 'record_train/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir_train):
    os.makedirs(record_dir_train)

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet50':
    G = resnet50()
    inc = 2048
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'bias' in key:
            params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, cosine=True, temp=args.T)
weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005,
                            nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005,
                            nesterov=True)
    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)

        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s.data.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.data.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.data.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)
        out1 = F1(output)
        loss = criterion(out1, target)
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        if not args.method == 'S+T':
            output = G(im_data_tu)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Loss T {:.6f} Method {}\n'.format(
                args.source, args.target,
                step, lr, loss.data, -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Method {}\n'.format(
                args.source, args.target,
                step, lr, loss.data, args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            test(target_loader_unl)
            G.train()
            F1.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(), os.path.join(args.checkpath,
                                                          "G_iter_model_{}_{}_to_{}_step_{}.pth.tar".format(
                                                              args.method, args.source, args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath, "F1_iter_model_{}_{}_to_{}_step_{}.pth.tar".format(
                               args.method, args.source, args.target, step)))

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(
        test_loss, correct, size,
        100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()
