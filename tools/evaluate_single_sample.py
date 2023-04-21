import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data

# import internal libs
import model
from tools.hessian import get_single_hessian, lanczos, get_single_hessian_single_layer
from torch.nn.utils import parameters_to_vector
from tools.utils import makedirs
from matplotlib import pyplot as plt


def save_evaluate_single_layer_hessian(args, eval_list, evec_list, filename=""):
    pth = os.path.join(args.save_path, "single_hessian")
    makedirs(pth)
    np.save(os.path.join(pth, f"hessian_evalue_{filename}.npy"), np.array(eval_list))
    np.save(os.path.join(pth, f"hessian_evector_{filename}.npy"), np.array(evec_list))


def save_evaluate_hessian(args, eval_list, evec_list, filename=""):
    pth = os.path.join(args.save_path, "hessian")
    makedirs(pth)
    np.save(os.path.join(pth, f"hessian_evalue_{filename}.npy"), np.array(eval_list))
    np.save(os.path.join(pth, f"hessian_evector_{filename}.npy"), np.array(evec_list))


def save_evaluate_confidence(args, conf_list, filename=""):
    pth = os.path.join(args.save_path, "confidence")
    makedirs(pth)
    np.save(os.path.join(pth, f"confidence_{filename}.npy"), np.array(conf_list))


def save_evaluate_single_layer_gradient(args, grad_list, norm_list, filename):
    pth = os.path.join(args.save_path, "single_gradient")
    makedirs(pth)
    np.save(os.path.join(pth, f"gradient_{filename}.npy"), np.array(grad_list))
    np.save(os.path.join(pth, f"gradient_norm_{filename}.npy"), np.array(norm_list))


def save_evaluate_gradient(args, grad_list, norm_list, filename=""):
    pth = os.path.join(args.save_path, "gradient")
    makedirs(pth)
    np.save(os.path.join(pth, f"gradient_{filename}.npy"), np.array(grad_list))
    np.save(os.path.join(pth, f"gradient_norm_{filename}.npy"), np.array(norm_list))


def evaluate_single_sample(args, train_set, model, criterion, mode="hessian", filename=""):
    """
    :param args: arguments for this project
    :param train_set: training dataset
    :param model: the neural network
    :param criterion: loss function
    :param mode: 'hessian': computing the eigenvalues and the eigenvectors using single sample
    'confidence': computing the confidence for each single sample
    'gradient': computing the gradient of weights computed using each single sample.
    'single_hessian': computing the eigenvalues and the eigenvectors for a single layer using a single sample
    'single_gradient': computing the gradients of weights in a single layer using a single sample
    :return:
    """
    assert mode in ["hessian", "confidence", "gradient", "single_hessian", "single_gradient"]
    if mode == "hessian" or mode == "single_hessian":
        eval_list = []
        evec_list = []
    elif mode == "confidence":
        conf_list = []
    elif mode == "gradient" or mode == "single_gradient":
        grad_list = []
        norm_list = []
    train_loader = Data.DataLoader(train_set, batch_size=1, shuffle=False)
    for step, (batch_X, batch_y) in enumerate(train_loader):
        if step >= args.eval_single:
            break
        model.zero_grad()
        if mode == "hessian":
            p = len(parameters_to_vector(model.parameters()))
            lam_get_hessian = lambda delta: get_single_hessian(args, model, criterion, batch_X, batch_y, delta)
            evals, evecs = lanczos(lam_get_hessian, p, neigs=6)
            eval_list.append(evals.numpy())
            evec_list.append(evecs.numpy())
        elif mode == "single_hessian":
            layer = model.layers[0]
            p = len(parameters_to_vector(layer.parameters()))
            lam_get_hessian = lambda delta: get_single_hessian_single_layer(args, model, layer, criterion, batch_X, batch_y, delta)
            evals, evecs = lanczos(lam_get_hessian, p, neigs=6)
            eval_list.append(evals.numpy())
            evec_list.append(evecs.numpy())
        elif mode == "confidence":
            output = model(batch_X)
            z = (output[0][batch_y[0].item()] - output[0][1 - batch_y[0].item()]).cpu().item()  # Just for 2-category classification
            conf_list.append(np.exp(z))
        elif mode == "gradient":
            output = model(batch_X)
            loss = criterion(output, batch_y)
            grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=False)
            grad = parameters_to_vector(grads)
            # grad_list.append(grad.detach().cpu().numpy())
            norm_list.append(grad.norm().item())
        elif mode == "single_gradient":
            layer = model.layers[0]
            output = model(batch_X)
            loss = criterion(output, batch_y)
            grads = torch.autograd.grad(loss, inputs=layer.parameters(), create_graph=False)
            grad = parameters_to_vector(grads)
            grad_list.append(grad.detach().cpu().numpy())
            norm_list.append(grad.norm().item())

    if mode == "hessian":
        save_evaluate_hessian(args, eval_list, evec_list, filename)
    elif mode == "single_hessian":
        save_evaluate_single_layer_hessian(args, eval_list, evec_list, filename)
    elif mode == "confidence":
        save_evaluate_confidence(args, conf_list, filename)
    elif mode == "gradient":
        save_evaluate_gradient(args, grad_list, norm_list, filename)
    elif mode == "single_gradient":
        save_evaluate_single_layer_gradient(args, grad_list, norm_list, filename)


if __name__ == '__main__':
    DATE1 = "2021-12-09"
    DATE2 = "2021-12-11"
    EPOCH = 150
    hessian_evalue_path = f"../tabular_result/outs/{DATE1}-commercial-lr0.1-adv_lr0.01-adv_step3-epoch400-seed2-zero/hessian/hessian_evalue_{EPOCH}_adv_data.npy"
    hessian_evector_path = f"../tabular_result/outs/{DATE1}-commercial-lr0.1-adv_lr0.01-adv_step3-epoch400-seed2-zero/hessian/hessian_evector_{EPOCH}_adv_data.npy"
    evalues = np.load(hessian_evalue_path)
    evectors = np.load(hessian_evector_path)

    evector_path = f"../tabular_result/models/{DATE2}-commercial-lr0.1-adv_lr0.01-adv_step3-epoch400-seed2-zero/evector/evector_{EPOCH}.pkl"
    evector = torch.load(evector_path).numpy()

    num = evalues.shape[0]
    evectors = evectors[:, :, 0]
    evector = evector[:, 0]

    cos_sims = []
    for i in range(num):
        cos_sim = np.dot(evector, evectors[i]) / (np.linalg.norm(evector) * np.linalg.norm(evectors[i]))
        cos_sims.append(cos_sim)
    cos_sims = np.array(cos_sims)

    plt.figure()
    plt.scatter(cos_sims, evalues[:, 0])
    plt.savefig(f"../tabular_result/outs/{DATE1}-commercial-lr0.1-adv_lr0.01-adv_step3-epoch400-seed2-zero/test.png")
    plt.close()
    print()
