from comet_ml import Experiment
import torch

from src.dataset import MCDataset
from src.model import GAE
from src.train import Trainer
from src.utils import calc_rmse, random_init, AverageMeter


def main(params, comet=False):
    if comet:
        experiment = Experiment(api_key="xK18bJy5xiPuPf9Dptr43ZuMk",
                        project_name="gcmc-ml100k", workspace="tanimutomo")
        experiment.log_parameters(params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(params['root'], params['dataset_name'])
    data = dataset[0].to(device)

    model = GAE(
        dataset.num_nodes,
        params['hidden_size'][0],
        params['hidden_size'][1],
        params['num_basis'],
        dataset.num_relations,
        int(data.num_users),
        params['drop_prob'],
        params['ster'],
        random_init,
        params['accum'],
        ).to(device)

    if comet:
        trainer = Trainer(model, dataset, data, calc_rmse, params['epochs'],
                params['lr'], params['weight_decay'], experiment)
    else:
        trainer = Trainer(model, dataset, data, calc_rmse,
                params['epochs'], params['lr'], params['weight_decay'])
    trainer.iterate()


if __name__ == '__main__':
    params = {
            'epochs': 1000,
            'lr': 0.01,
            'weight_decay': 0,
            'ster': 1e-3,
            'drop_prob': 0.7,
            'accum': 'stack',

            'hidden_size': [500, 75],
            'num_basis': 2,

            'root': 'data/ml-100k',
            'dataset_name': 'ml-100k'
            }
    # main(params, comet=True)
    main(params)

