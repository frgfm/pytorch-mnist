# Classic architectures on MNIST
This repository is an ongoing implementation of different basic DL architectures for Image classification.
The implementation is in PyTorch with Visdom support for visualization.

## Installation
This project was only developed in Python 3.6 using PyTorch 0.4.1. If you happen to be running a superior version, depending on your edits, you might encounter issue during runtime (breaking changes between Pytorch version).
```
git clone https://github.com/frgfm/pytorch-mnist.git
cd pytorch-mnist
pip install requirements.txt
```

## Usage

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

![visdom_loss](static/images/lenet5_training.gif)



If you wish to resume a training, use the --resume flag

```bash
python main.py 10 --lr 5e-5 --momentum 0.9 --weight_decay 5e-4 -n --batch_size 8 --gpu 0 --resume Lenet5_checkpoint_best.pth.tar
```

## TODO
- [x] LeNet5 implementation
- [ ] Resuming from checkpoint
- [ ] Different MLP combinations (+ regularization)
- [ ] Stop criterion
