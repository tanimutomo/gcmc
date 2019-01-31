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


class MCDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(MCDataset, self).__init__(root, transform, pre_transform)
        # processed_path[0]は処理された後のデータで，process methodで定義される
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_nodes(self):
        return self.data.x.shape[0]

    @property
    def raw_file_names(self):
        return ['u1.base', 'u1.test']

    @property
    def processed_file_names(self):
        return 'data.pt'

    
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
        train_df, train_nums = self.create_df(train_csv)
        test_df, test_nums = self.create_df(test_csv)

        train_idx, train_gt = self.create_gt_idx(train_df, train_nums)
        test_idx, test_gt = self.create_gt_idx(test_df, train_nums)

        train_df['item_id'] = train_df['item_id'] + train_nums['user']

        x = torch.arange(train_nums['node'], dtype=torch.long)

        edge_user = torch.tensor(train_df['user_id'].values)
        edge_item = torch.tensor(train_df['item_id'].values)
        edge_index = torch.stack((torch.cat((edge_user, edge_item), 0),
                                  torch.cat((edge_item, edge_user), 0)), 0)
        edge_index = edge_index.to(torch.long)

        edge_type = torch.tensor(train_df['relation'])
        edge_type = torch.cat((edge_type, edge_type), 0)

        edge_norm = copy.deepcopy(edge_index[1])
        for idx in range(train_nums['node']):
            count = (train_df == idx).values.sum()
            edge_norm = torch.where(edge_norm==idx,
                                    torch.tensor(count),
                                    edge_norm)

        edge_norm = (1 / edge_norm.to(torch.float))


        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_norm = edge_norm
        data.train_idx = train_idx
        data.test_idx = test_idx
        data.train_gt = train_gt
        data.test_gt = test_gt
        data.num_users = torch.tensor([train_nums['user']])
        data.num_items = torch.tensor([train_nums['item']])
        
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    
    def create_df(self, csv_path):
        col_names = ['user_id', 'item_id', 'relation', 'ts']
        df = pd.read_csv(csv_path, sep='\t', names=col_names)
        df = df.drop('ts', axis=1)
        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
        df['relation'] = df['relation'] - 1

        nums = {
                'user': df.max()['user_id'] + 1,
                'item': df.max()['item_id'] + 1,
                'node': df.max()['user_id'] + df.max()['item_id'] + 2,
                'edge': len(df)
                }

        return df, nums


    def create_gt_idx(self, df, nums):
        df['idx'] = df['user_id'] * nums['item'] + df['item_id']
        idx = torch.tensor(df['idx'])
        gt = torch.tensor(df['relation'])

        return idx, gt

    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data[0]

        
    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(root='./data/ml-100k', name='ml-100k')
    data = dataset[0]
    print(data)
    data = data.to(device)
