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

from architectures.lenet5 import LeNet5

parser = argparse.ArgumentParser()
parser.add_argument("nb_epoch", help="Enter the number of epochs you wish to train")
parser.add_argument("--lr", help="Learning rate (default: 2e-3)")
parser.add_argument("--momentum", "-m", help="SGD Momentum (default: 0.9)")
parser.add_argument("--batch_size", "-b", help="Batch size (default: 4)")
parser.add_argument("--gpu", help="GPU ID you wish to use (default: 0)")
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    nb_epoch_it = len(train_loader)

    # Architecture choice
    net = LeNet5()
    torch.cuda.set_device(int(args.gpu))
    net.cuda()

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 2e-3
    if args.lr is not None:
        lr = float(args.lr)
    momentum = 0.9
    if args.momentum is not None:
        momentum = float(args.momentum)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # Visdom
    plot_opts = dict(showlegend=True,
                     width=600, height=600, ytype='log',
                     title='MNIST Training (%s)' % net.name(),
                     xlabel='Batch index', ylabel='Loss')

    def train(net, it_idx=None, checkpoint_freq=1000, checkpoint_folder='checkpoint'):

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
            # Checkpoint
            if (batch_idx + 1) % checkpoint_freq == 0 or (batch_idx + 1) == len(train_loader):
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                save_path = os.path.join(checkpoint_folder, net.name() + '_checkpoint_iter%s.pth' % (it_idx + batch_idx))
                torch.save(net.state_dict(), save_path)

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

    # Loop over the entire dataset
    for epoch in range(int(args.nb_epoch)):
        print('\nEpoch %s/%s' % (epoch + 1, int(args.nb_epoch)))
        # Training
        training_loss = train(net, it_idx=epoch * nb_epoch_it + 1)
        # Validation
        if epoch == 0:
            vis.line(X=[0], Y=[validation_loss],
                     win=train_win, opts=plot_opts, name='Validation loss',
                     update='append')
        validation_loss, accuracy = test(net)
        vis.line(X=[(epoch + 1) * nb_epoch_it], Y=[validation_loss],
                 win=train_win, opts=plot_opts, name='Validation loss',
                 update='append')
        print('Training loss: %s, Validation loss: %s, Accuracy: %s' % (training_loss, validation_loss, accuracy))


if __name__ == '__main__':
    main()
