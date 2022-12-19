import pandas as pd
import numpy as np
import plotly.express as px


class WineDataset:

    def __init__(self, path):
        df = pd.read_csv(path)
        df = df.replace(['white', 'red'], [0, 1]).values
        df[:,:-1] = self.normalization(df[:,:-1])
        df[:,-1] = df[:,-1] > 5
        train, test = self.__devide_data(df)
        self.train_input, self.train_target, self.test_input, self.test_target = train[:, :-1], train[:, -1],\
                                                                                 test[:,:-1], test[:,-1]

    @staticmethod
    def normalization(array):
        return (array - np.min(array, axis=0)) / (np.max(array, axis=0) - np.min(array, axis=0))

    @staticmethod
    def __devide_data(array):
        l = len(array)
        ind = np.arange(l)
        ind_prm = np.random.permutation(ind)
        train_ind = ind_prm[:np.int32(0.8 * l)]
        valid_ind = ind_prm[np.int32(0.8 * l):]
        train = array[train_ind]
        valid = array[valid_ind]
        return train, valid

    def __call__(self):
        return {'train_input': self.train_input,
                'train_target': self.train_target,
                'test_input': self.test_input,
                'test_target': self.test_target
                }

