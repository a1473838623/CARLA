import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.mypath import MyPath
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PSM(Dataset):
    """
    PSM Dataset.
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('psm'), train=True, transform=None, sanomaly=None, mean_data=None,
                 std_data=None):
        super(PSM, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []

        # Window size and stride configuration
        wsz, stride = 200, 10  # Common settings for PSM, adjustable

        # Load Train Data for Scalar Fit
        train_df = pd.read_csv(os.path.join(self.root, 'train.csv'))
        train_data = train_df.values[:, 1:]  # Drop timestamp
        train_data = np.nan_to_num(train_data)

        scaler = StandardScaler()
        scaler.fit(train_data)

        if self.train:
            self.mean = scaler.mean_
            self.std = scaler.scale_
            data = train_data
            # PSM train set usually doesn't have labels provided in standard benchmarks, assumed all normal
            # However, if labels exist, they should be loaded. Standard PSM assumes unsupervised train.
            labels = np.zeros(len(data))
        else:
            self.mean, self.std = mean_data, std_data
            test_df = pd.read_csv(os.path.join(self.root, 'test.csv'))
            test_data = test_df.values[:, 1:]
            test_data = np.nan_to_num(test_data)

            # Use provided scaler stats or re-fit if not provided (though test should use train stats)
            if mean_data is None:
                data = scaler.transform(test_data)
            else:
                # Manually apply scaling if mean/std passed from train set
                data = (test_data - self.mean) / self.std

            # Load Labels
            test_label_df = pd.read_csv(os.path.join(self.root, 'test_label.csv'))
            labels = test_label_df.values[:, 1:]
            labels = np.nan_to_num(labels).flatten()  # Ensure 1D array

        self.targets = np.asarray(labels)
        self.data = np.asarray(data)
        self.data, self.targets = self.convert_to_windows(wsz, stride)

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0] - w_size) / stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st:st + w_size]
            # Window label is 1 if any point in window is anomalous
            if np.sum(self.targets[st:st + w_size]) > 0:
                lbl = 1
            else:
                lbl = 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

    def __getitem__(self, index):
        ts_org = torch.from_numpy(self.data[index]).float().to(device)

        if len(self.targets) > 0:
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long).to(device)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts_org.shape[0], ts_org.shape[1])
        out = {'ts_org': ts_org, 'target': target,
               'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}
        return out

    def get_ts(self, index):
        return self.data[index]

    def get_info(self):
        return self.mean, self.std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")