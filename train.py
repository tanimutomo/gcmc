import torch

from dataset import MCDataset
from model import GAE
from utils import nll_loss


class Trainer:
    def __init__(self, root, dataset_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root = root
        self.dataset_name = dataset_name
        self.hidden_size = [500, 75]
        self.num_basis = 2
        self.drop_prob = 0.7

    def create_dataset(self):
        # self.dataset = MCDataset(root='./data/ml-100k', name='ml-100k')
        self.dataset = MCDataset(self.root, self.dataset_name)
        # self.num_train_users = dataset.num_train_users
        # self.num_train_items = dataset.num_train_items
        # self.num_test_users = dataset.num_test_users
        # self.num_test_items = dataset.num_test_items

        self.train_data, self.test_data = self.dataset[0][0], self.dataset[1][0]
        self.train_data = self.train_data.to(device)
        self.test_data = self.test_data.to(device)

    def training_setting(self):
        model = GAE(
                dataset.num_train_users,
                self.hidden_size[0]
                self.hidden_size[1]
                self.num_basis,
                dataset.num_relations,
                self.drop_out
                )

        criterion = nll_loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.005)

    def iterate(self):
        pass

    def train(self):
        model.train()

        out = model(
                self.train_data.x,
                self.train_data.edge_index,
                self.train_data.edge_type,
                self.train_data.edge_norm
                )


    def test(self):
        pass

    def summary(self):
        pass


