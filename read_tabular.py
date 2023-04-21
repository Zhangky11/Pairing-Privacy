import sys
import os
import logging
import argparse
import torch

# sys.path.append("/home/huangjin/adversarial_hessian/tabular_experiment")
import dataset
from config import DATE, MOMENT, DATASETS, INITS, ABS_POSES, TRAIN_ARGS, VFUNCS


def add_args():
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser() # (description = "Code for reference-value experiment (binary classfication)")
    parser.add_argument("--dataset", default = "census", type = str,
                        choices = DATASETS,
                        help = "set the dataset used.")
    # set the path for data
    parser.add_argument('--data_path', default = '/home/zhangky/data/tabular', type = str, 
                        help = "path for dataset.")  
    parser.add_argument('--device', default = 0, type = int,
                        help = "set the device.")

    args = parser.parse_args()
    if TRAIN_ARGS[args.dataset]["if_fix"]:
        #logger.warning("Due to experiment settings for {} have been fixed, following args will be changed ->".format(args.dataset))
        for key in TRAIN_ARGS[args.dataset].keys():
            if key != "if_fix":
                #logger.warning("args.{} \twill be changed from {} to {}.".format(key, args.__dict__[key], TRAIN_ARGS[args.dataset][key]))
                args.__dict__[key] = TRAIN_ARGS[args.dataset][key]

    return args  

def read_tabular():
    args = add_args()
    if args.dataset == "credit":
        X_train, y_train, X_test, y_test, X, y, train_loader = dataset.load_credit(args)
    elif args.dataset == "abalone":
        X_train, y_train, X_test, y_test, X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, train_loader = dataset.load_abalone(args)
    elif args.dataset == "census" or args.dataset == "commercial":
        # dataset_info, X_train, y_train, X_test, y_test, X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, train_loader = dataset.load_census(args)
        dataset_info, X_train, y_train, X_test, y_test, X_train_sampled, y_train_sampled, \
        X_test_sampled, y_test_sampled, train_loader, test_loader = dataset.load_tabular(args)
    return X_train, y_train, X_test, y_test,train_loader, test_loader

if __name__ == "__main__":
    X_train, y_train, X_test, y_test,train_loader, test_loader = read_tabular()
    # print(X_train)
    print("X_train shape: ", X_train.shape)
    print("Y_train shape: ", y_train.shape)
    print("X_train: ", X_train)
    print("y_train: ", y_train)
    print("X_test.shape: ", X_test.shape)
    print("y_test: ", y_test.shape)
    