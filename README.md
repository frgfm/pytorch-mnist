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

### Running the visdom server
Start the visdom server to visualize your training loss:
```bash
python -m visdom.server
```
By default, visdom server will start on port 8097, so navigating to http://localhost:8097 will allow you to see live training results.

If you happen to perform the training on a remote server, you will need to first allow external connection to this port:
```bash
sudo ufw allow 8097
python -m visdom.server
```
Then locally, navigate to `http://<REMOTE_SERVER_IP>:8097` for live training results.


### Training your model
Then open another terminal and run the following command to start training:
```bash
python main.py 10 --lr 5e-5 --momentum 0.9 --weight_decay 5e-4 -n --batch_size 8 --gpu 0
```
You can choose the number of epochs, the learning rate, the momentum, the weight decay, whether you wish to use nesterov momentum, the batch size as well as the GPU to use for your training.

<p align="center"><img align="center" src="https://github.com/frgfm/pytorch_mnist/blob/master/images/lenet5_training.gif" width="400" /></p>

If you wish to resume a training, use the --resume flag
```bash
python main.py 10 --lr 5e-5 --momentum 0.9 --weight_decay 5e-4 -n --batch_size 8 --gpu 0 --resume Lenet5_checkpoint_best.pth.tar
```

## TODO
- [x] LeNet5 implementation
- [ ] Resuming from checkpoint
- [ ] Different MLP combinations (+ regularization)
- [ ] Stop criterion
