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
        # self.train_loss_meter = AverageMeter()
        # self.train_rmse_meter = AverageMeter()
        # self.test_rmse_meter = AverageMeter()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)

    def iterate(self):
        for epoch in range(self.epochs):
            loss, train_rmse = self.train(epoch)
            # self.train_loss_meter.update(loss.item())
            # self.train_rmse_meter.update(train_rmse.item())
            # if epoch % 10 == 0:
            test_rmse = self.test()
            # self.test_rmse_meter.update(test_rmse.item())
            self.summary(epoch, loss, train_rmse, test_rmse)
            # else:
            #     self.summary(epoch, loss)
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

        # print('--------Parameter---------')
        # for param in self.model.parameters():
        #     print(param.grad)
        # print('--------------------------')

        # if epoch % 10 == 0:
        rmse = self.calc_rmse(out[self.data.train_idx], self.data.train_gt)
        return loss.item(), rmse.item()
        # else:
        #     return loss, None


    def test(self):
        self.model.eval()
        out = self.model(
                self.data.x, self.data.edge_index, 
                self.data.edge_type, self.data.edge_norm
                )

        rmse = self.calc_rmse(out[self.data.test_idx], self.data.test_gt)

        return rmse.item()


    def summary(self, epoch, loss, train_rmse=None, test_rmse=None):
        # min_loss = {'value': 1e+10, 'epoch': 0}
        # min_train_rmse = {'value': 1e+10, 'epoch': 0}
        # min_test_rmse = {'value': 1e+10, 'epoch': 0}
        # for min_metric in list(min_loss, min_train_rmse, min_test_rmse):
        #     if min_metric['value'] < loss:
        #         min_metric['value'] = loss
        #         min_metric['epoch'] = epoch

        if test_rmse is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, loss))
        else:
            # print('')
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
                epoch, self.epochs, loss, train_rmse, test_rmse))
            # print('')
            
        

if __name__ == '__main__':
    pass
