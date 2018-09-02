# Classic architectures on MNIST
This repository is an ongoing implementation of different basic DL architectures for Image classification.
The implementation is in PyTorch with Visdom support for visualization.


## How to use it
Add your architectures (compatible with MNIST 28x28 images) in the architectures folder, and change the net definition accordingly.
Start the visdom server to visualize your training loss:
```bash
python -m visdom.server
```

Then just run the following command to start training:
```bash
python main.py
```

## TODO
- [x] LeNet5 implementation
- [] Different MLP combinations (+ regularization)
- [] Stop criterion
- [] Advanced Visdom visualization
- [] Resuming from checkpoint