import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time


class Trainer:
    def __init__(self, model, dataset, data, calc_rmse,
                 optimizer, experiment=None):
        self.model = model
        self.dataset = dataset
        self.data = data
        self.calc_rmse = calc_rmse
        self.optimizer = optimizer
        self.experiment = experiment

    def training(self, epochs):
        self.epochs = epochs
        iter_f_e = []
        iter_f_d = []
        iter_b = []
        for epoch in range(self.epochs):
            loss, train_rmse = self.train_one(epoch, iter_f_e, iter_f_d, iter_b)
            test_rmse = self.test()
            self.summary(epoch, loss, train_rmse, test_rmse)
            if self.experiment is not None:
                metrics = {'loss': loss,
                           'train_rmse': train_rmse,
                           'test_rmse': test_rmse}
                self.experiment.log_metrics(metrics, step=epoch)
            if epoch > 0 and epoch % 100 == 0:
                print("{}\t{}\t{}".format(np.mean(iter_f_e[3:]), np.mean(iter_f_d[3:]), np.mean(iter_b[3:])))

        print('END TRAINING')

    def train_one(self, epoch, iter_f_e, iter_f_d, iter_b):
        self.model.train()
        out = self.model(self.data.x, self.data.edge_index,
                         self.data.edge_type, self.data.edge_norm,
                         iter_f_e, iter_f_d)
        loss = F.cross_entropy(out[self.data.train_idx], self.data.train_gt)

        t1 = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        iter_b.append(time.time() - t1)

        rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
        return loss.item(), rmse.item()

    def test(self):
        self.model.eval()
        a = []
        b = []
        out = self.model(self.data.x, self.data.edge_index, 
                         self.data.edge_type, self.data.edge_norm, a, b)
        rmse = self.calc_rmse(out[self.data.test_idx], self.data.test_gt)
        return rmse.item()

    def summary(self, epoch, loss, train_rmse=None, test_rmse=None):
        if test_rmse is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
                epoch, self.epochs, loss, train_rmse, test_rmse))
