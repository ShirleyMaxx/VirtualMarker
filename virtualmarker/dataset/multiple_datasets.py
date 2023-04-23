import random
import numpy as np
from torch.utils.data.dataset import Dataset


class MultipleDatasets(Dataset):
    def __init__(self, dbs):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])

    def __len__(self):
        return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        for i in range(self.db_num):
            if index < self.db_len_cumsum[i]:
                db_idx = i
                break
        if db_idx == 0:
            data_idx = index
        else:
            data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]