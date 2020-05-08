from comet_ml import Experiment
import torch
import yaml

from dataset import MCDataset
from model import GAE
from trainer import Trainer
from utils import calc_rmse, ster_uniform, random_init, init_xavier, init_uniform, Config


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
    with open('config.yml') as f:
        config = yaml.safe_load(f)
    main(config)
    # main(config, comet=True)