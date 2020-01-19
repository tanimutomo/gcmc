from comet_ml import Experiment
import torch

from src.dataset import MCDataset
from src.model import GAE
from src.train import Trainer
from src.utils import calc_rmse, ster_uniform, random_init, init_xavier, init_uniform, Config


def main(config, comet=False):
    config = Config(config)

    # comet-ml setting
    if comet:
        experiment = Experiment(api_key=config.api_key,
                        project_name=config.project_name, workspace=config.workspace)
        experiment.log_parameters(config)

    # device and dataset setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(config.root, config.dataset_name)
    data = dataset[0].to(device)

    # add some params to config
    config.num_nodes = dataset.num_nodes
    config.num_relations = dataset.num_relations
    config.num_users = int(data.num_users)

    # set and init model
    model = GAE(config, random_init).to(device)
    model.apply(init_xavier)

    # train
    if comet:
        trainer = Trainer(model, dataset, data, calc_rmse, config.epochs,
                config.lr, config.weight_decay, experiment)
    else:
        trainer = Trainer(model, dataset, data, calc_rmse,
                config.epochs, config.lr, config.weight_decay)
    trainer.iterate()


if __name__ == '__main__':
    config = {
        # train setting
        'epochs': 1,
        'lr': 0.01,
        'weight_decay': 0,
        'drop_prob': 0.7,

        # network
        'accum': 'split_stack',
        'hidden_size': [500, 75],
        'num_basis': 2,
        'rgc_bn': True,
        'rgc_relu': True,
        'dense_bn': True,
        'dense_relu': True,
        'bidec_drop': False,

        # dataset
        'root': 'data/ml-100k',
        'dataset_name': 'ml-100k',

        # comet-ml (optional)
        'workspace': '',
        'project_name': '',
        'api_key': ''
    }
    # main(config, comet=True)
    main(config)

