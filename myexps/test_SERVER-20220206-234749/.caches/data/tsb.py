from sklearn.model_selection import train_test_split
import scipy.io as sio
from torch.utils import data
import numpy as np

def tsb_split(filename = './data/tsb/g1.mat'):
	X, y = load_tishby_toy_dataset(filename)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, train_size = 0.9)

	return X_train, X_test, y_train, y_test

class tsbData(data.Dataset):
	def __init__(self, X, y):
		self.data = X.astype(np.float32)
		self.targets_ori = y.astype(np.int64)
		self.targets = self.targets_ori.reshape(-1)

	def __getitem__(self, index):
		return self.data[index], self.targets[index], index

	def __len__(self):
		return len(self.data)

def load_tishby_toy_dataset(filename, assign_random_labels=False, seed=42):
    np.random.seed(seed)
    
    data = sio.loadmat(filename)
    F = data['F']
    
    if assign_random_labels:
        y = np.random.randint(0, 2)
    else:
        y = data['y'].T
    
    return F, y

def get_datasets(data_path, batch_size):
	X_train, X_test, y_train, y_test = tsb_split()

	train_set = tsbData(X_train, y_train)
	test_set = tsbData(X_test, y_test)

	return train_set, test_set