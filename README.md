# Classic architectures on MNIST
This repository is an ongoing implementation of different basic DL architectures for Image classification.
The implementation is in PyTorch with Visdom support for visualization.

## Requirements
The torch, torchvision and visdom packages are required to properly use the repo.
Tested on the following version:
```python
import sys
import torch, torchvision, visdom
print('Python %s' % '.'.join(map(str, sys.version_info[:3])))
print('PyTorch %s, Torchvision %s, Visdom %s' % (torch.__version__, torchvision.__version__, visdom.__version__))
```
```console
Python 3.6.6
PyTorch 0.4.1, Torchvision 0.2.1, Visdom 0.1.8.5
```


## How to use it
Add your architectures (compatible with MNIST 28x28 images) in the architectures folder, and change the net definition accordingly.
Start the visdom server to visualize your training loss:
```bash
python -m visdom.server
```

Then open another terminal and run the following command to start training:
```bash
python main.py 3 --lr 2e-3 --momentum 0.9 --batch_size 4
```
You can choose the number of epochs, the learning rate, the momentum and the batch size for your training.

<p align="center"><img align="center" src="https://github.com/frgfm/pytorch_mnist/blob/master/images/lenet5_traning.gif" width="600" /></p>

## TODO
- [x] LeNet5 implementation
- [ ] Different MLP combinations (+ regularization)
- [ ] Stop criterion
- [ ] Advanced Visdom visualization
- [ ] Resuming from checkpoint