# train classifier

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import os
from nets import *
from utils.misc import model_snapshot


def classifier(embedding, train_data, val_data, writer,
               epochs=30,
               cuda=True,
               log_interval=100,
               test_interval=1,
               lr=0.1,
               logdir='log/default'):
    assert isinstance(embedding, nn.Module), 'Embedding is not a module'
    model = Classifier(embedding)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    n_iter = 0
    if cuda:
        model.cuda()

    # optimizer
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    best_acc,old_file = 0, None
    t_begin = time.time()
    try:
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, label) in enumerate(train_data):
                index_label = label.clone()
                if cuda:
                    data, label = data.cuda(), label.cuda()
                data, label = Variable(data), Variable(label)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizer.step()
                n_iter += 1

                if batch_idx % log_interval == 0:
                    pred = output.data.max(1)[1]  # get the index of the log-probability
                    correct = pred.cpu().eq(index_label).sum()
                    acc = correct * 1.0 / len(data)
                    writer.add_scalar('classifier_acc', acc, n_iter)
                    writer.add_scalar('classifier_loss', loss.data[0], n_iter)
                    print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(train_data.dataset),
                        loss.data[0], acc
                    ))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_data)
            eta = speed_epoch * epochs - elapse_time
            print('Elapsed {:.2f}s, {:.2f}s/epoch, {:.2f}s/batch, ets{:.2f}s'.format(
                elapse_time, speed_epoch, speed_batch, eta))
            model_snapshot(model, os.path.join(logdir, 'latest.pth'))

            if epoch % test_interval == 0:
                model.eval()
                test_loss = 0
                correct = 0
                for data, label in val_data:
                    index_label = label.clone()
                    if cuda:
                        data, label = data.cuda(), label.cuda()
                    data, label = Variable(data, volatile=True), Variable(label)
                    output = model(data)
                    test_loss += F.cross_entropy(output, label).data[0]
                    pred = output.data.max(1)[1]
                    correct += pred.cpu().eq(index_label).sum()

                test_loss = test_loss / len(val_data)
                acc = 100. * correct / len(val_data.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(val_data.dataset), acc))
                if acc > best_acc:
                    new_file = os.path.join(logdir, 'best-{}.pth'.format(epoch))
                    model_snapshot(model, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    old_file = new_file
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))








