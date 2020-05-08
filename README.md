# Graph Convolutional Matrix Completion (Pytorch)
Re-implementation of [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (PyTorch and PyTorch Geometric)

![overview](./figs/overview.png)

![approach](./figs/approach.png)

## Note
This repository is **NOT** an official implementation of the paper.  
The official implementation is [this](https://github.com/riannevdberg/gc-mc) (Tensorflow).  
Our experimental result is shown below and it doesn't reach to the score of the original.

## Setup
- Setup a virtual environment with python 3.6 or newer
- Install requirements (pip)
  ```
  pip install -r requirements/1.txt
  pip install --verbose --no-cache-dir -r requirements/2.txt
  pip install -r requirements/3.txt
  ```
Installation of Pytorch Geometric is very troublesome and may destroy your python environment.  
So, we strongly recommend to use a virtual environment (e.g. pyenv, virtualenv, pipenv, etc.).  
Please see [Pytorch Geometirc official document](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) for more details.  


## Train and Test
```
cd src
python train.py
```
- Configuration:  
The settings for train and test are in `config.yml`.  

- Dataset:  
Training dataset is MovieLens-100k.
The dataset is automatically downloaded in `data/` by running `src/train.py`.


## Results
Note that this repo doesn't reach to the original one.

| | Test RMSE |
|:--|--:|
| Ours | 0.968 |
| Original | 0.910 |
