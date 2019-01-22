import os
import copy
import glob
import shutil
import pandas as pd
import numpy as np

import torch
from torch_scatter import scatter_add
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import one_hot

# from utils import extract_zip

class MCDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(MCDataset, self).__init__(root, transform, pre_transform)
        # processed_path[0]は処理された後のデータで，process methodで定義される
        self.train_data, self.train_slices = torch.load(self.processed_paths[0])
        self.test_data, self.test_slices = torch.load(self.processed_paths[1])
        
    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_train_users(self):
        return self.proc_train_info['num_users']

    @property
    def num_test_users(self):
        return self.proc_test_info['num_users']

    @property
    def num_train_items(self):
        return self.proc_train_info['num_items']

    @property
    def num_test_items(self):
        return self.proc_test_info['num_items']

    @property
    def raw_file_names(self):
        return ['u1.base', 'u1.test']

    @property
    def processed_file_names(self):
        return ['data_0.pt', 'data_1.pt']

    
    def download(self):
        if self.name == 'ml-100k':
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        path = download_url(url, self.root)
        extract_zip(path, self.raw_dir, self.name)
        os.unlink(path)
        for file in glob.glob(os.path.join(self.raw_dir, self.name, '*')):
            shutil.move(file, self.raw_dir)
        os.rmdir(os.path.join(self.raw_dir, self.name))

        
    def process(self):
        train_csv, test_csv = self.raw_paths
        self.proc_train_info, proc_train_data = self.preprocess(train_csv)
        self.proc_test_info, proc_test_data = self.preprocess(test_csv)

        # for key, data in self.proc_train_info.items():
        #     print(type(data))
        # for key, data in self.proc_train_data.items():
        #     print(type(data))
        
        train_data = Data(x=proc_train_data['x'], 
                edge_index=proc_train_data['edge_index'])
        # train_data.num_users = torch.tensor(proc_train_info['num_users'])
        # train_data.num_items = torch.tensor(proc_train_info['num_items'])
        # train_data.num_nodes = proc_train_info['num_nodes']
        # train_data.num_edges = proc_train_info['num_edges']
        train_data.edge_type = proc_train_data['edge_type']
        train_data.edge_norm = proc_train_data['edge_norm']
        
        train_data, train_slices = self.collate([train_data])
        torch.save((train_data, train_slices), self.processed_paths[0])


        test_data = Data(x=proc_test_data['x'], 
                edge_index=proc_test_data['edge_index'])
        # test_data.num_users = torch.tensor(proc_test_info['num_users'])
        # test_data.num_items = torch.tensor(proc_test_info['num_items'])
        # test_data.num_nodes = proc_test_info['num_nodes']
        # test_data.num_edges = proc_test_info['num_edges']
        test_data.edge_type = proc_test_data['edge_type']
        test_data.edge_norm = proc_test_data['edge_norm']
        
        test_data, test_slices = self.collate([test_data])
        torch.save((test_data, test_slices), self.processed_paths[1])

    
    def preprocess(self, csv_path):
        col_names = ['user_id', 'item_id', 'relation', 'ts']
        raw_data = pd.read_csv(csv_path, sep='\t', names=col_names)

        num_users = raw_data.max()['user_id']
        num_items = raw_data.max()['item_id']
        num_nodes = num_users + num_items
        num_edges = len(raw_data)

        proc_data = raw_data.drop('ts', axis=1)
        proc_data['user_id'] = proc_data['user_id'] - 1
        proc_data['item_id'] = proc_data['item_id'] + num_users - 1

        x = torch.arange(num_nodes, dtype=torch.long)

        edge_user = torch.tensor(proc_data['user_id'].values)
        edge_item = torch.tensor(proc_data['item_id'].values)
        edge_index = torch.stack((torch.cat((edge_user, edge_item), 0),
                                  torch.cat((edge_item, edge_user), 0)), 0)
        edge_index = edge_index.to(torch.long)

        edge_type = torch.tensor(proc_data['relation'])
        edge_type = torch.cat((edge_type, edge_type), 0)

        edge_norm = copy.deepcopy(edge_index[1])
        for idx in range(num_nodes):
            count = (proc_data == idx).values.sum()
            edge_norm = torch.where(edge_norm==idx,
                                    torch.tensor(count),
                                    edge_norm)

        edge_norm = (1 / edge_norm.to(torch.double))

        return [
                {
                    'num_users': num_users, 
                    'num_items': num_items,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges
                },
                {
                    'x': x,
                    'edge_index': edge_index,
                    'edge_type': edge_type,
                    'edge_norm': edge_norm
                }
                ]

    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir,
            'data_{}.pt'.format(idx)))
        return data

        
    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(root='./data/ml-100k', name='ml-100k')
    train_data, test_data = dataset[0][0], dataset[1][0]
    print(train_data)
    print(test_data)
    train_data, test_data = train_data.to(device), test_data.to(device)
