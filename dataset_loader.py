import numpy as np
import os
from pymongo import MongoClient
from gridfs import GridFS
import json


class DatasetLoader(object):

    fnames = {'airlines': ['airlines', 'regression'],
              'airlines_classification': ['airlines_classification', 'classification'],
              'airlines_classification10k': ['airlines_classification10k', 'classification'],
              'mnist': ['mnist', 'multiclass'],
              'higgs': ['higgs', 'binary'],
              'boston': ['bostonHousing', 'regression'],
              'power': ['power-plant', 'regression'],
              'concrete': ['concrete', 'regression'],
              'energy': ['energy', 'regression'],
              'kin8nm': ['kin8nm', 'regression'],
              'naval': ['naval-propulsion-plant', 'regression'],
              'protein': ['protein-tertiary-structure', 'regression'],
              'wine_red': ['wine-quality-red', 'regression'],
              'yacht': ['yacht', 'regression'],
              'year': ['YearPredictionMSD', 'classification'],
              'australian': ['Australian', 'classification'],
              'breast': ['Breast', 'classification'],
              'crabs': ['Crabs', 'classification'],
              'iono': ['Iono', 'classification'],
              'pima': ['Pima', 'classification'],
              'sonar': ['Sonar', 'classification'],
              'glass': ['glass', 'multiclass'],
              'new-thyroid': ['new-thyroid', 'multiclass'],
              'svmguide2': ['svmguide2', 'multiclass'],
              'satellite': ['satellite', 'multiclass'],
              'vehicle': ['vehicle', 'multiclass'],
              'waveform': ['waveform', 'multiclass'],
              'wine': ['wine', 'multiclass'],
              'vowel': ['vowel', 'multiclass']
            }

    def __init__(self, name, data_path='/data/', from_database=False, db_chain="mongodb://"):
        self.data_path = data_path + '/'
        self.name = DatasetLoader.fnames[name][0]
        self.type = DatasetLoader.fnames[name][1]
        assert self.type in ['regression', 'classification', 'multiclass']
        self.from_database = from_database
        if self.from_database:
            client = MongoClient(db_chain)
            db = client.datasets
            self.splits = db[self.name]
            self.fs = GridFS(db)

    def file_path(self):
        return self.data_path + self.name + '/data/data.txt'

    def read_data(self):
        if self.type == "multiclass":
            data = np.load("%s/%s/data.npz" % (self.data_path, self.name))
            
            X, Y = data['arr_0'], data['arr_1'][:, None]
            
            data = {'X': X, 'Y': Y.astype(np.int)}
        elif self.from_database:
            out = self.fs.find_one({"filename": f"{self.name}.json"})
            data_json = json.loads(out.read())

            if self.type is 'regression':
                data = {'X': np.array(data_json['X']), 'Y': np.array(data_json['y'])}
            else:
                data = {'X': np.array(data_json['X']), 'Y': np.array(data_json['y']).astype(np.int)}
        else:
            data = np.loadtxt(self.file_path())

            xindexfile = '{}{}/data/index_features.txt'.format(self.data_path, self.name)
            yindexfile = '{}{}/data/index_target.txt'.format(self.data_path, self.name)
            xindices = np.loadtxt(xindexfile, dtype=np.int)
            yindex = np.loadtxt(yindexfile, dtype=np.int)

            X = data[:, xindices]
            y = data[:, yindex]
            y = y.reshape([y.shape[0], 1])
            
            if self.type is 'regression':
                data = {'X': X, 'Y': y}
            else:
                data = {'X': X, 'Y': y.astype(np.int)}

        return data

    def get_data(self, split=0):
        if not self.from_database:
            path = self.file_path()
            if self.type != "multiclass" and not os.path.isfile(path):
                raise IOError("Error when loading dataset {}, directory {} not found".format(
                                self.name, path))

        full_data = self.read_data()
        split_data = self.split(full_data, split)
        split_data = self.normalize(split_data, 'X')

        if self.type is 'regression':
            split_data = self.normalize(split_data, 'Y')
        else:
            split_data.update({'Y_mean': 0.0})
            split_data.update({'Y_std': 1.0})

        return split_data

    def split(self, full_data, split):
        if self.type == "multiclass":
            folds = np.load("%s/%s/folds.npz" % (self.data_path, self.name))
            
            itrain, itest = folds['arr_0'], folds['arr_1']
            
            index_train = itrain[split, :]
            index_test = itest[split, :]

        elif self.name == "higgs" or self.name == "airlines_classification10k":

            n_test = int(1e4)
            n = full_data['Y'].shape[0]
            end_train = int(n - n_test)

            np.random.seed(split)

            permutation = np.random.choice(range(n), n, replace = False)
            
            end_test = n
            
            index_train = permutation[ 0 : end_train ]
            index_test = permutation[ end_train : n ]

        elif self.from_database:
            index_test = json.loads(self.splits.test.find_one({'split': split})['index'])
            index_train = json.loads(self.splits.train.find_one({'split': split})['index'])
        else:
            train_ind_file = self.data_path + self.name + '/data/index_train_' + str(split) + '.txt'
            test_ind_file = self.data_path + self.name + '/data/index_test_' + str(split) + '.txt'
            index_train = np.loadtxt(train_ind_file, dtype=np.int)
            index_test = np.loadtxt(test_ind_file, dtype=np.int)

        X_train = full_data['X'][index_train, :]
        y_train = full_data['Y'][index_train, :]
        X_test = full_data['X'][index_test, :]
        y_test = full_data['Y'][index_test, :]

        return {'X': X_train, 'Xs': X_test, 'Y': y_train, 'Ys': y_test}

    def normalize(self, split_data, X_or_Y):
        m = np.average(split_data[X_or_Y], 0)[None, :]
        s = np.std(split_data[X_or_Y], 0)[None, :] + 1e-6

        split_data[X_or_Y] = (split_data[X_or_Y] - m) / s
        if X_or_Y == 'X':
            split_data[X_or_Y + 's'] = (split_data[X_or_Y + 's'] - m) / s

        split_data.update({X_or_Y + '_mean': m.flatten()})
        split_data.update({X_or_Y + '_std': s.flatten()})

        return split_data
