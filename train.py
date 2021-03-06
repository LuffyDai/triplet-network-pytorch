from __future__ import print_function
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_mnist_loader import MNIST_t
from triplet_image_loader import TripletImageLoader
from tripletnet import Tripletnet
# from visdom import Visdom
from tensorboardX import SummaryWriter
from logbook import Logger
from nets import *
import numpy as np
from classifier_train import classifier
from datasets import get_Dataset
from triplet_datasets import get_TripletDataset
from losses import TripletLossSoftmax
from context import Context


logger = Logger('triplet-net')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--log', default='log', type=str,
                    help='filename to log')
parser.add_argument('--name', default='cifar10', type=str,
                    help='name of experiment')
parser.add_argument('--net', default='CIFARNet', type=str,
                    help='name of network to use')
parser.add_argument('--use-fc', default=False,
                    help='use last fc layer')

best_acc = 0
n_iter = 0


def main():

    global args, best_acc, writer
    args = parser.parse_args()
    writer = SummaryWriter(comment='_' + args.name + '_triplet_network')
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # global plotter
    # plotter = VisdomLinePlotter(env_name=args.name)

    kwargs = {'num_workers': 1 if args.name == 'stl10' else 4,
              'pin_memory': True} if args.cuda else {}  # change num_workers from 1 to 4

    train_triplet_loader, test_triplet_loader, train_loader, test_loader = \
        get_TripletDataset(args.name, args.batch_size, **kwargs)

    cmd = "model=%s()" % args.net
    local_dict = locals()
    exec(cmd, globals(), local_dict)
    model = local_dict['model']
    print(args.use_fc)
    if not args.use_fc:
        tnet = Tripletnet(model)
    else:
        tnet = Tripletnet(Classifier(model))
    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    time_string = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    log_directory = "runs/%s/" % (time_string + '_' + args.name)

    with Context(os.path.join(log_directory, args.log), parallel=True):
        for epoch in range(1, args.epochs + 1):
            # train for one epoch
            train(train_triplet_loader, tnet, criterion, optimizer, epoch)
            # evaluate on validation set
            acc = test(test_triplet_loader, tnet, criterion, epoch)

            # remember best acc and save checkpoint
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': tnet.state_dict(),
                'best_prec1': best_acc,
            }, is_best)

        checkpoint_file = 'runs/%s/'%(args.name) + 'model_best.pth.tar'
        assert os.path.isfile(checkpoint_file), 'Nothing to load...'
        checkpoint_cl = torch.load(checkpoint_file)
        cmd = "model_cl=%s()" % args.net
        exec(cmd, globals(), local_dict)
        model_cl = local_dict['model_cl']
        if not args.use_fc:
            tnet = Tripletnet(model_cl)
        else:
            tnet = Tripletnet(Classifier(model_cl))
        tnet.load_state_dict(checkpoint_cl['state_dict'])
        classifier(tnet.embeddingnet if not args.use_fc else tnet.embeddingnet.embedding,
                   train_loader, test_loader, writer, logdir=log_directory)

    writer.close()

def train(train_loader, tnet, criterion, optimizer, epoch):

    global n_iter

    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, ((data1, label1), (data2, label2), (data3, label3)) in enumerate(train_loader):

        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)

        # loss_triplet = criterion(dista, distb)
        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd
        if args.name == 'svhn' or 'stl10':
            loss = loss_triplet

        n_iter += 1

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))


        #label_batch = Variable(label1, requires_grad=False).long()
        #writer.add_embedding(embedded_x.data, metadata=label_batch.data, global_step=n_iter)
        #writer.add_embedding(embedded_y, metadata=label2.data, label_img=data2.data, global_step=n_iter)
        #writer.add_embedding(embedded_z, metadata=label3.data, label_img=data3.data, global_step=n_iter)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg,
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))
            writer.add_scalar('train_loss', loss_triplet.data[0], n_iter)
            writer.add_scalar('train_acc', acc, n_iter)
    # log avg values to somewhere

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, ((data1, label1), (data2, label2), (data3, label3)) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        #test_loss = criterion(dista, distb).data[0]
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))
        if batch_idx == 0:
            label = label1
            out = embedded_x.data
        else:
            label = torch.cat((label, label1), 0)
            out = torch.cat((out, embedded_x.data), 0)

    label_batch = Variable(label, requires_grad=False).long()

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    writer.add_scalar('test_loss', losses.avg, epoch)
    writer.add_scalar('test_acc', accs.avg, epoch)
    if epoch == 1 or epoch == args.epochs:
        writer.add_embedding(out, metadata=label_batch.data, global_step=epoch)

    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()
