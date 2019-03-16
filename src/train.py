import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, dataset, data, calc_rmse,
            epochs, lr, weight_decay, experiment=None):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.model = model
        self.dataset = dataset
        self.data = data
        self.calc_rmse = calc_rmse
        self.experiment = experiment

        self.train_setting()


    def train_setting(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)

    def iterate(self):
        for epoch in range(self.epochs):
            loss, train_rmse = self.train(epoch)
            test_rmse = self.test()
            self.summary(epoch, loss, train_rmse, test_rmse)
            if self.experiment is not None:
                metrics = {
                        'loss': loss,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        }
                self.experiment.log_metrics(metrics, step=epoch)

        print('END TRAINING')


    def train(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(
                self.data.x, self.data.edge_index,
                self.data.edge_type, self.data.edge_norm
                )
        loss = self.criterion(out[self.data.train_idx], self.data.train_gt)
        loss.backward()
        self.optimizer.step()

        rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
        return loss.item(), rmse.item()


    def test(self):
        self.model.eval()
        out = self.model(
                self.data.x, self.data.edge_index, 
                self.data.edge_type, self.data.edge_norm
                )

        rmse = self.calc_rmse(out[self.data.test_idx], self.data.test_gt)

        return rmse.item()


    def summary(self, epoch, loss, train_rmse=None, test_rmse=None):
        if test_rmse is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
                epoch, self.epochs, loss, train_rmse, test_rmse))
            
        
