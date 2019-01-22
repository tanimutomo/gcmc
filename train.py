import torch
import torch.nn as nn

from dataset import MCDataset
from model import GAE
from utils import calc_rmse


class Trainer:
    def __init__(self, root, dataset_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.root = root
        self.dataset_name = dataset_name
        self.hidden_size = [500, 75]
        self.num_basis = 2
        self.drop_prob = 0.7
        self.epochs = 100

    def set_dataset(self):
        pass
        # self.dataset = MCDataset(root='./data/ml-100k', name='ml-100k')
        # self.num_train_users = dataset.num_train_users
        # self.num_train_items = dataset.num_train_items
        # self.num_test_users = dataset.num_test_users
        # self.num_test_items = dataset.num_test_items

        # self.train_data, self.test_data = self.dataset[0][0], self.dataset[1][0]
        # self.train_data = self.train_data.to(self.device)
        # self.test_data = self.test_data.to(self.device)

    def train_setting(self):
        dataset = MCDataset(self.root, self.dataset_name)
        print(dataset[0])
        self.data = dataset[0].to(self.device)
        self.model = GAE(
                dataset.num_nodes,
                self.hidden_size[0],
                self.hidden_size[1],
                self.num_basis,
                dataset.num_relations,
                self.data.num_users,
                self.drop_prob
                )
        self.model = self.model.to(self.device)

        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=0.01, weight_decay=0.005)


    def iterate(self):
        for epoch in range(self.epochs):
            loss, train_rmse = self.train()
            if epoch % 10 == 0:
                test_rmse = self.test()
                self.summary(epoch, loss, train_rmse, test_rmse)
            else:
                self.summary(epoch, loss, train_rmse)

        print('END TRAINING')


    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(
                self.data.x, self.data.edge_index,
                self.data.edge_type, self.data.edge_norm
                )
        loss = self.criterion(out[self.data.train_idx], self.data.train_gt)
        loss.backward()
        optimizer.step()

        rmse = calc_rmse(out[self.data.train_idx], self.data.train_gt)

        return loss.item(), rmse.item()


    def test(self):
        model.eval()
        out = self.model(
                self.data.x, self.data.edge_index, 
                self.data.edge_type, self.data.edge_norm
                )

        rmse = calc_rmse(out[self.data.test_idx], self.data.test_gt)

        return rmse.item()


    def summary(self, epoch, loss, train_rmse, test_rmse=None):
        if test_rmse is not None:
            print('[ Epoch: {} / Loss: {} / RMSE: {} ]'.format(
                epoch, loss, train_rmse))
        else:
            print('')
            print('[ Epoch: {} / Loss: {} / RMSE: {} / Test RMSE: {} ]'.format(
                epoch, loss, train_rmse, test_rmse))
            print('')
            
        

if __name__ == '__main__':
    root = 'data/ml-100k'
    name = 'ml-100k'
    trainer = Trainer(root, name)
    trainer.train_setting()
    trainer.iterate()
