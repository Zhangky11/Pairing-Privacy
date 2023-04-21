import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
import model
from tools.plot import plot_curves
from .train_logistic_regression import train_logistic_regression_model, eval_logistic_regression_model
from .train_classification import train_classification_model, eval_classification_model, train_classification_model_hessian
from .train_classification import adv_train_classification_model_hessian
from .train_regression import train_regression_model, eval_regression_model
# from logger import logger


def train_model(args, X_train, y_train, X_test, y_test, train_loader, task="classification"):
    if task == "logistic_regression":
        return train_logistic_regression_model(args, X_train, y_train, X_test, y_test, train_loader)
    elif task == "classification":
        return train_classification_model(args, X_train, y_train, X_test, y_test, train_loader)
    elif task == "classification_hessian":
        return train_classification_model_hessian(args, X_train, y_train, X_test, y_test, train_loader)
    elif task == "adv_classification_hessian":
        return adv_train_classification_model_hessian(args, X_train, y_train, X_test, y_test, train_loader)
    elif task == "regression":
        return train_regression_model(args, X_train, y_train, X_test, y_test, train_loader)
    else:
        raise NotImplementedError(f"Unknown task: {task}.")




def load_model(args, model_path, X_train, y_train, X_test, y_test):
    """load the pretrained models
    """
    in_dims = X_train.size(-1)
    out_dims = len(set(y_train.detach().cpu().numpy()))
    if "logistic" in args.arch:
        if args.dataset in ["census", "commercial"]:
            net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=1)
    else:
        if args.dataset in ["census", "commercial"] \
                + [f"commercial_rule{i}_classification" for i in range(1, 11)] \
                + [f"commercial_rule{i}_classification_ind" for i in range(1, 11)] \
                + [f"gaussian_rule{i}_classification_ind" for i in range(1, 11)] \
                + [f"gaussian_8d_rule{i}_classification" for i in range(1, 11)]:
            net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=out_dims)
        elif args.dataset in ["bike"] \
                + [f"commercial_rule{i}_regression" for i in range(1, 11)]:
            net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=1)
        else:
            raise Exception
    print(net)
    net = net.float().to(args.device)

    net.load_state_dict(torch.load(model_path, map_location=torch.device(f"cuda:{args.device}")))
    print("The model has existed in model path '{}'. Load pretrained model.".format(model_path))
    if "logistic" in args.arch:
        if args.dataset in ["census", "commercial"]:
            net = eval_logistic_regression_model(net, X_test, y_test, nn.BCEWithLogitsLoss())
    else:
        if args.dataset in ["census", "commercial"] + [f"commercial_rule{i}_classification" for i in range(1, 11)]:
            net = eval_classification_model(net, X_test, y_test, nn.CrossEntropyLoss())
        elif args.dataset in ["bike", "commercial_rule1_regression"]:
            net = eval_regression_model(net, X_test, y_test, nn.MSELoss())

    return net