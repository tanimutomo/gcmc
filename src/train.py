import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, dataset, data, calc_rmse, epochs, lr, weight_decay):
        self.model = model
        self.dataset = dataset
        self.data = data
        self.calc_rmse = calc_rmse

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_setting()


    def train_setting(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)

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
        loss.backward()
        self.optimizer.step()

        # print('--------Parameter---------')
        # for param in self.model.parameters():
        #     print(param.grad)
        # print('--------------------------')

        # if epoch % 10 == 0:
        rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
        return loss, rmse
        # else:
        #     return loss, None


    def test(self):
        self.model.eval()
        out = self.model(
                self.data.x, self.data.edge_index, 
                self.data.edge_type, self.data.edge_norm
                )

        rmse = self.calc_rmse(out[self.data.test_idx], self.data.test_gt)

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
    pass
