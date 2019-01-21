import os
import pandas as pd
import numpy as np
import torch
from torch_scatter import scatter_add

from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar)
from torch_geometric.utils import one_hot

class MCDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(MCDataset, self).__init__(root, transform, pre_transform)
        self.name = name
        # processed_path[0]は処理された後のデータで，process methodで定義される
        self.train_data, self.train_slices = torch.load(self.processed_paths[0])
        self.test_data, self.test_slices = torch.load(self.processed_paths[1])
        
    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def raw_file_names(self):
        return ['u1.base', 'u1.test']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'test_data.pt']

    
    def download(self):
        if self.name == 'ml-100k':
            url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        path = download_url(url, self.root)
        extract_tar(path, self.raw_dir)
        os.unlink(path)

        
    def process(self):
        train_csv, test_csv = self.raw_paths
        train_info, train_data = self.preprocess(train_csv)
        test_info, test_data = self.preprocess(test_csv)
        
        train_data = Data(x=train_data['x'], 
                edge_index=train_data['edge_index'])
        train_data.num_users = train_data['num_users']
        train_data.num_items = train_data['num_items']
        train_data.num_nodes = train_data['num_nodes']
        train_data.num_edges = train_data['num_edges']
        train_data.edge_type = train_data['edge_type']
        train_data.edge_norm = train_data['edge_norm']
        
        train_data, train_slices = self.collate([train_data])
        torch.save((train_data, train_slices), self.processed_paths[0])


        test_data = Data(x=test_data['x'], 
                edge_index=test_data['edge_index'])
        test_data.num_users = test_data['num_users']
        test_data.num_items = test_data['num_items']
        test_data.num_nodes = test_data['num_nodes']
        test_data.num_edges = test_data['num_edges']
        test_data.edge_type = test_data['edge_type']
        test_data.edge_norm = test_data['edge_norm']
        
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

    
    def get(self, data_type):
        data = torch.load(os.path.join(self.processed_dir,
            '{}_data.pt'.format(data_type)))
        return data

        
    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)
        
