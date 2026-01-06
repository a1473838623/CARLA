import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.mypath import MyPath
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NIPS_TS_Water(Dataset):
    """
    NIPS_TS_Water Dataset.
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('nips_ts_water'), train=True, transform=None, sanomaly=None,
                 mean_data=None, std_data=None):
        super(NIPS_TS_Water, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        wsz, stride = 100, 10

        # Load Train Data to fit Scaler
        train_data_path = os.path.join(self.root, "NIPS_TS_Water_train.npy")
        train_data_raw = np.load(train_data_path)

        scaler = StandardScaler()
        scaler.fit(train_data_raw)

        if self.train:
            self.mean = scaler.mean_
            self.std = scaler.scale_
            data = scaler.transform(train_data_raw)
            # Assuming train data has no anomalies or we don't have train labels file loaded here
            labels = np.zeros(len(data))
        else:
            self.mean, self.std = mean_data, std_data
            test_data_path = os.path.join(self.root, "NIPS_TS_Water_test.npy")
            test_data_raw = np.load(test_data_path)

            if mean_data is None:
                data = scaler.transform(test_data_raw)
            else:
                data = (test_data_raw - self.mean) / self.std

            label_path = os.path.join(self.root, "NIPS_TS_Water_test_label.npy")
            labels = np.load(label_path)

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