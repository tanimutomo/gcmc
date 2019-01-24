import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.drop_prob = 0.3
        self.epochs = 1000
        self.lr = 0.01
        self.ster = 1e-3


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
                int(self.data.num_users),
                self.drop_prob,
                self.ster
                )
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=0.005)


    def iterate(self):
        for epoch in range(self.epochs):
            loss, train_rmse = self.train(epoch)
            # if epoch % 10 == 0:
            test_rmse = self.test()
            self.summary(epoch, loss, train_rmse, test_rmse)
            # else:
            #     self.summary(epoch, loss)

        print('END TRAINING')


    def train(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(
                self.data.x, self.data.edge_index,
                self.data.edge_type, self.data.edge_norm
                )
        loss = self.criterion(out[self.data.train_idx], self.data.train_gt)
        # loss = F.nll_loss(out[self.data.train_idx], self.data.train_gt)
        loss.backward()
        self.optimizer.step()

        # print('--------Parameter---------')
        # for param in self.model.parameters():
        #     print(param.grad)
        # print('--------------------------')

        # if epoch % 10 == 0:
        rmse = calc_rmse(out[self.data.train_idx], self.data.train_gt)
        return loss, rmse
        # else:
        #     return loss, None

        # print('model grad: ', list(self.model.parameters())[0].grad)
        # print(self.model.bidec.basis_matrix.grad)
        # print(self.model.bidec.coefs[2].grad)

    def test(self):
        self.model.eval()
        out = self.model(
                self.data.x, self.data.edge_index, 
                self.data.edge_type, self.data.edge_norm
                )

        rmse = calc_rmse(out[self.data.test_idx], self.data.test_gt)

        return rmse


    def summary(self, epoch, loss, train_rmse=None, test_rmse=None):
        if test_rmse is None:
            print('[ Epoch: {}/{} | Loss: {} ]'.format(
                epoch, self.epochs, loss))
        else:
            # print('')
            print('[ Epoch: {}/{} | Loss: {} | RMSE: {} | Test RMSE: {} ]'.format(
                epoch, self.epochs, loss, train_rmse, test_rmse))
            # print('')
            
        

if __name__ == '__main__':
    root = 'data/ml-100k'
    name = 'ml-100k'
    trainer = Trainer(root, name)
    trainer.train_setting()
    trainer.iterate()
