import os
import os.path as osp
import copy
import random

# import ML libs
import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# import internal libs
from tools.utils import label_encode, recognize_features_type, set_discrete_continuous
from config import USED_SAMPLE_NUM, SPLIT_RATIO


def load_census(args):
    """prepare the credit dataset including shuffle and splitting.
        Extraction was done by Barry Becker from the 1994 Census database. Detailed description could be found in http://archive.ics.uci.edu/ml/datasets/Census+Income

        Args:
            args (dict): set containing all program arguments.

        Returns:
            train set, test set, the whole X & y and train_loader.
        """
    data_folder = osp.join(args.data_path, args.dataset)
    X_train = np.load(osp.join(data_folder, "X_train.npy"))
    y_train = np.load(osp.join(data_folder, "y_train.npy"))
    X_test = np.load(osp.join(data_folder, "X_test.npy"))
    y_test = np.load(osp.join(data_folder, "y_test.npy"))
    X_train_sampled = np.load(osp.join(data_folder, "X_train_sampled.npy"))
    y_train_sampled = np.load(osp.join(data_folder, "y_train_sampled.npy"))
    X_test_sampled = np.load(osp.join(data_folder, "X_test_sampled.npy"))
    y_test_sampled = np.load(osp.join(data_folder, "y_test_sampled.npy"))

    X_train = torch.from_numpy(X_train).float().to(args.device)
    y_train = torch.from_numpy(y_train).long().to(args.device)
    X_test = torch.from_numpy(X_test).float().to(args.device)
    y_test = torch.from_numpy(y_test).long().to(args.device)
    X_train_sampled = torch.from_numpy(X_train_sampled).float().to(args.device)
    y_train_sampled = torch.from_numpy(y_train_sampled).long().to(args.device)
    X_test_sampled = torch.from_numpy(X_test_sampled).float().to(args.device)
    y_test_sampled = torch.from_numpy(y_test_sampled).long().to(args.device)

    # create dataloader
    train_set = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    with open(osp.join(data_folder, "info.json"), "r") as f:
        dataset_info = json.load(f)

    return dataset_info,\
           X_train, y_train, X_test, y_test,\
           X_train_sampled, y_train_sampled,\
           X_test_sampled, y_test_sampled,\
           train_loader




# def load_census(args):
#     """prepare the credit dataset including shuffle and splitting.
#     Extraction was done by Barry Becker from the 1994 Census database. Detailed description could be found in http://archive.ics.uci.edu/ml/datasets/Census+Income
#
#     Args:
#         args (dict): set containing all program arguments.
#
#     Returns:
#         train set, test set, the whole X & y and train_loader.
#     """
#     # Read Dataset
#     df = pd.read_csv(os.path.join(args.data_path, 'adult.csv'), delimiter=',', skipinitialspace=True)
#
#     # Remove useless columns
#     del df['fnlwgt']
#     del df['education-num']
#
#     # Remove Missing Values
#     for col in df.columns:
#         if '?' in df[col].unique():
#             df[col][df[col] == '?'] = df[col].value_counts().index[0]  # TODO: This code set '?' to a default value, but reasonable?
#
#     # Features Categorization
#     columns = df.columns.tolist()
#     columns = columns[-1:] + columns[:-1]  # put the label in the '0-th' dim
#     df = df[columns]
#     class_name = 'class'  # the col-name of the target
#     attribute_names = columns[1:]
#
#     type_features, _ = recognize_features_type(df)
#
#     discrete, _ = set_discrete_continuous(columns, type_features, class_name)  # find the discrete/continuous fields
#
#     columns_tmp = list(columns)  # TODO: unused?
#     columns_tmp.remove(class_name)  # TODO: unused?
#
#     # Dataset Preparation for Scikit Alorithms
#     df_le, _ = label_encode(df, discrete)
#     X = df_le.loc[:, df_le.columns != class_name].values  # numpy array (32561, 12)
#     y = df_le[class_name].values  # numpy array (32561,)
#
#     # split the dataset
#     # Their shape -- (26048, 12) (6513, 12) (26048,) (6513,)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATIO[args.dataset], random_state=args.seed)
#
#     logger.info("- > train shape: {}; test shape: {}".format(X_train.shape, X_test.shape))
#     logger.debug("sample of X - > \n{}".format(X_train[:1]))
#     logger.debug("sample of y - > \n{}".format(y_train[:10]))
#
#     # Sample 1000 X and y from train  # TODO: used for what?
#     idx_lst = list(range(X_train.shape[0]))
#     random.shuffle(idx_lst)
#     X = X_train[idx_lst][:USED_SAMPLE_NUM[args.dataset]]
#     y = y_train[idx_lst][:USED_SAMPLE_NUM[args.dataset]]
#     logger.info("- > sampled dataset shape: {}".format(X.shape))
#
#     # prepare the data
#     X_train, X_test = torch.from_numpy(X_train).float().to(args.device), torch.from_numpy(X_test).float().to(args.device)
#     y_train, y_test = torch.from_numpy(y_train).long().to(args.device), torch.from_numpy(y_test).long().to(args.device)
#     X, y = torch.from_numpy(X).float().to(args.device), torch.from_numpy(y).long().to(args.device)
#
#     # create dataloader
#     train_set = Data.TensorDataset(X_train, y_train)
#     train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
#
#     # prepare the data
#
#     dataset_info = {
#         "attributes": attribute_names,
#         "target": class_name,
#         "n_attribute": X_train.shape[1],
#         "n_train_sample": X_train.shape[0],
#         "n_test_sample": X_test.shape[0],
#         "n_sampled": X.shape[0]
#     }
#
#
#     return dataset_info, X_train, y_train, X_test, y_test, X, y, train_loader


def get_census_attributes(data_path):
    with open(osp.join(data_path, "census", "info.json"), "r") as f:
        dataset_info = json.load(f)

    # # Read Dataset
    # df = pd.read_csv(os.path.join(data_path, 'adult.csv'), delimiter=',', skipinitialspace=True)
    #
    # # Remove useless columns
    # del df['fnlwgt']
    # del df['education-num']
    #
    # # Features Categorization
    # columns = df.columns.tolist()
    # columns = columns[-1:] + columns[:-1]  # put the label in the '0-th' dim
    # attribute_names = columns[1:]

    return dataset_info["attributes"]