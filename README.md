# gcmc
PyTorch and PyTorch geometric based implementation of [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263).

This repository is NOT an official implementation of that paper.
The official implementation is [this](https://github.com/riannevdberg/gc-mc). It is Tensorflow based implementation.

## Training
### Docker (Recommend)
- Install docker and docker-compose (see docker official document)
- Clone this repository
```
git clone https://github.com/tanimutomo/gcmc.git
```
- Build the docker container
```
docker-compose -f ./docker/docker-compose-{cpu/gpu}.yml build
```
- Run main.py 
```
docker-compose -f ./docker/docker-compose-{cpu/gpu}.yml run experiment python3 main.py
```

### Local
#### Install requirements
Installation of Pytorch Geometric is difficult and can destroy your local python environment.  
So, if you already installed docker and docker-compose, recommend to use docker (above).  
Please see [Pytorch Geometirc document](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) for more details.  
- torch==1.0.0
- torchvision
- comet_ml (you must have comet-ml account_)
- torch-scatter
- torch-sparse
- torch-cluster
- torch-spline_conv
- torch-geometric
