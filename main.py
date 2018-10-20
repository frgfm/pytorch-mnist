#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Benchmark of deep learning architectures for classification on MNIST
'''

__author__ = 'François-Guillaume Fernandez'
__license__ = 'MIT License'
__version__ = '0.1'
__maintainer__ = 'François-Guillaume Fernandez'
__status__ = 'Development'

import sys
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from visdom import Visdom
import argparse
import shutil

from architectures.lenet5 import LeNet5

PTH_CHECKPOINT_FOLDER = 'checkpoint'

parser = argparse.ArgumentParser()
parser.add_argument("nb_epoch", type=int, help="Enter the number of epochs you wish to train")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
parser.add_argument("--momentum", "-m", type=float, default=0., help="SGD Momentum (default: 0)")
parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay (default: 0)")
parser.add_argument("--nesterov", "-n", action='store_true', help="Nesterov momentum (default: False)")
parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size (default: 4)")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID you wish to use (default: 0)")
parser.add_argument("--start_epoch", type=int, default=0, help="manual epoch number (default: 0)")
parser.add_argument("--workers", type=int, default=2, help="Number of workers used for data loading (default: 2)")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint file to resume (default: None)")
args = parser.parse_args()


def progress(count, total, status=''):
    """
    Display a progress bar
    Args:
        count (int): number of iterations already processed
        total (int): total number of iterations
        status (str): status information you want to print out
    """

    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * (filled_len - 1) + '>' + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s - %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, os.path.join(PTH_CHECKPOINT_FOLDER, filename + '.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(PTH_CHECKPOINT_FOLDER, filename + '.pth.tar'),
                        os.path.join(PTH_CHECKPOINT_FOLDER, filename + '_best.pth.tar'))


def main():

    # Connect to local Visdom server
    vis = Visdom(port=8097, use_incoming_socket=False)

    # Image set
    # Normalize tensors (MNIST mean & std)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Train & test sets
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 4
    if args.batch_size is not None:
        batch_size = int(args.batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.workers)

    nb_epoch_it = len(train_loader)

    # Net
    if not os.path.exists(PTH_CHECKPOINT_FOLDER):
        os.makedirs(PTH_CHECKPOINT_FOLDER)

    # Architecture choice
    net = LeNet5()
    torch.cuda.set_device(int(args.gpu))
    net.cuda()

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # Resuming
    if args.resume:
        if os.path.exists(os.path.join(PTH_CHECKPOINT_FOLDER, args.resume)):
            checkpoint = torch.load(os.path.join(PTH_CHECKPOINT_FOLDER, args.resume))
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Resuming checkpoint %s (epoch %s)' % (args.resume, args.start_epoch))
        else:
            raise ValueError('Unable to locate checkpoint %s' % args.resume)

    # Visdom
    plot_opts = dict(showlegend=True,
                     width=900, height=600, ytype='log',
                     title='MNIST Training (%s)' % net.name(),
                     xlabel='Batch index', ylabel='Loss')

    def train(net, it_idx=None, checkpoint_freq=1000):

        net.train()
        for batch_idx, (x, target) in enumerate(train_loader, 0):
            # Work with tensors on GPU
            x, target = x.cuda(), target.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward & optimize
            outputs = net.forward(x)
            loss = criterion(outputs, target)

            # Evaluation logging
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                # Command line status
                status = 'Training loss: %s' % loss.item()
                progress(batch_idx + 1, nb_epoch_it, status)
                # Visdom plot
                if 'train_win' not in globals():
                    global train_win
                    train_win = vis.line(X=[it_idx + batch_idx], Y=[loss.item()],
                                         opts=plot_opts, name='Training loss')
                else:
                    vis.line(X=[it_idx + batch_idx], Y=[loss.item()],
                             win=train_win, opts=plot_opts, name='Training loss',
                             update='append')
            # # Checkpoint
            # if (batch_idx + 1) % checkpoint_freq == 0 or (batch_idx + 1) == len(train_loader):
            #     save_path = os.path.join(PTH_CHECKPOINT_FOLDER, net.name() + '_checkpoint_iter%s.pth' % (it_idx + batch_idx))
            #     torch.save(dict(epoch=, arch=net.name(), state_dict=net.state_dict(), 'optimizer': optimizer.state_dict()),
            #                save_path)

            loss.backward()
            optimizer.step()

        return loss.item()

    def test(net):

        net.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for x, target in test_loader:
                # Work with tensors on GPU
                x, target = x.cuda(), target.cuda()

                # Forward + Backward & optimize
                outputs = net.forward(x)
                loss += criterion(outputs, target).item()
                # Index of max log-probability
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(test_loader)
        accuracy = correct / float(batch_size * len(test_loader))

        return loss, accuracy

    # Get initial validation loss
    validation_loss, accuracy = test(net)
    best_acc1 = accuracy
    is_best = False

    # Loop over the entire dataset
    for epoch in range(int(args.nb_epoch)):
        print('\nEpoch %s/%s' % (args.start_epoch + epoch + 1, args.start_epoch + int(args.nb_epoch)))
        # Training
        training_loss = train(net, it_idx=(args.start_epoch + epoch) * nb_epoch_it + 1)
        # Validation
        if epoch == 0:
            vis.line(X=[args.start_epoch * nb_epoch_it], Y=[validation_loss],
                     win=train_win, opts=dict(markers=True, showlegend=True), name='Validation loss',
                     update='append')
        validation_loss, accuracy = test(net)
        vis.line(X=[(args.start_epoch + epoch + 1) * nb_epoch_it], Y=[validation_loss],
                 win=train_win, opts=dict(markers=True, showlegend=True), name='Validation loss',
                 update='append')
        print('Training loss: %s, Validation loss: %s, Accuracy: %s' % (training_loss, validation_loss, accuracy))
        # Checkpoint
        if accuracy > best_acc1:
            best_acc1 = accuracy
            is_best = True
        save_checkpoint(dict(epoch=args.start_epoch + epoch + 1, arch=net.name(), state_dict=net.state_dict(), optimizer=optimizer.state_dict()),
                        is_best, '%s_checkpoint' % net.name())


if __name__ == '__main__':
    main()
