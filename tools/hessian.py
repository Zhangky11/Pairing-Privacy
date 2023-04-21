import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import time
from tqdm import tqdm
import torch.nn.functional as F
from tools.utils import makedirs
import pickle
import dataset
from torch.autograd.functional import hessian
from itertools import combinations
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh
from matplotlib import pyplot as plt


def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec).cpu()

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()


def get_hessian(args, model, criterion, train_loader, vector):
    p = len(parameters_to_vector(model.parameters()))
    hvp = torch.zeros(p).to(args.device)
    n = len(train_loader.dataset)
    vector = vector.to(args.device)
    for step, (batch_X, batch_y) in enumerate(train_loader):
        output = model(batch_X)
        loss = criterion(output, batch_y) / n
        grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads_2 = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads_2)
    return hvp


def get_hessian_single_layer(args, model, layer, criterion, train_loader, vector):
    p = len(parameters_to_vector(layer.parameters()))
    hvp = torch.zeros(p).to(args.device)
    n = len(train_loader.dataset)
    vector = vector.to(args.device)
    for step, (batch_X, batch_y) in enumerate(train_loader):
        output = model(batch_X)
        loss = criterion(output, batch_y) / n
        grads = torch.autograd.grad(loss, inputs=layer.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads_2 = [g.contiguous() for g in torch.autograd.grad(dot, layer.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads_2)
    return hvp


def get_hessian_on_x(args, model, criterion, x, y, vector):
    p = x.size(1)
    hvp = torch.zeros(p).to(args.device)
    vector = vector.to(args.device)
    output = model(x)
    loss = criterion(output, y)
    grads = torch.autograd.grad(loss, inputs=x, create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads_2 = [g.contiguous() for g in torch.autograd.grad(dot, x, retain_graph=True)]
    hvp += parameters_to_vector(grads_2)
    return hvp


def get_single_hessian(args, model, criterion, x, y, vector):
    p = len(parameters_to_vector(model.parameters()))
    hvp = torch.zeros(p).to(args.device)
    vector = vector.to(args.device)
    output = model(x)
    loss = criterion(output, y)
    grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads_2 = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
    hvp += parameters_to_vector(grads_2)
    return hvp


def get_single_hessian_single_layer(args, model, layer, criterion, x, y, vector):
    p = len(parameters_to_vector(layer.parameters()))
    hvp = torch.zeros(p).to(args.device)
    vector = vector.to(args.device)
    output = model(x)
    loss = criterion(output, y)
    grads = torch.autograd.grad(loss, inputs=layer.parameters(), create_graph=True)
    dot = parameters_to_vector(grads).mul(vector).sum()
    grads_2 = [g.contiguous() for g in torch.autograd.grad(dot, layer.parameters(), retain_graph=True)]
    hvp += parameters_to_vector(grads_2)
    return hvp

