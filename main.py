import torch

from src.dataset import MCDataset
from src.model import GAE
from src.train import Trainer
from src.utils import calc_rmse, random_init


def main():
    hidden_size = [500, 75]
    num_basis = 2
    drop_prob = 0.7
    epochs = 1000
    lr = 0.02
    weight_decay = 0
    ster = 1e-3

    root = 'data/ml-100k'
    name = 'ml-100k'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(root, name)
    data = dataset[0].to(device)

    model = GAE(
        dataset.num_nodes,
        hidden_size[0],
        hidden_size[1],
        num_basis,
        dataset.num_relations,
        int(data.num_users),
        drop_prob,
        ster,
        random_init
        ).to(device)

    trainer = Trainer(model, dataset, data, calc_rmse, epochs, lr, weight_decay)
    trainer.iterate()


if __name__ == '__main__':
    main()
