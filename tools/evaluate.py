import os

# import ML libs
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.utils.data as Data

# import internal libs
import model
from tools.hessian import get_single_hessian, lanczos, get_single_hessian_single_layer
from torch.nn.utils import parameters_to_vector
from tools.utils import makedirs
from tools.evaluate_single_sample import evaluate_single_sample
from visualize.eval_lambda_plot import compare_single_sample_lambda_max, compare_different_samples, single_sample_lambda_hist


def evaluate(args, net, X_train, y_train, criterion):
    single_layer_evals = []
    all_layer_evals = []
    num = 5
    for chosen_epoch in range(0, args.epoch, 50):
        # net.load_state_dict(torch.load(os.path.join(args.model_path, "model", f"model_{chosen_epoch}.pt")))
        # adv_dataset = torch.load(os.path.join(args.model_path, "adv_dataset", f"adv_dataset_{chosen_epoch}.pt"))
        # single_hessian_evals = np.load(os.path.join(args.save_path, "single_hessian", f"hessian_evalue_{chosen_epoch}_adv_data.npy"))
        # single_hessian_evecs = np.load(os.path.join(args.save_path, "single_hessian", f"hessian_evector_{chosen_epoch}_adv_data.npy"))
        single_gradient_norm = np.load(os.path.join(args.save_path, "single_gradient", f"gradient_norm_{chosen_epoch}_adv_data.npy"))
        confidence = np.load(os.path.join(args.save_path, "confidence", f"confidence_{chosen_epoch}_adv_data.npy"))
        gradient_norm = np.load(os.path.join(args.save_path, "gradient", f"gradient_norm_{chosen_epoch}_adv_data.npy"))
        single_gradient = np.load(os.path.join(args.save_path, "single_gradient", f"gradient_{chosen_epoch}_adv_data.npy"))
        # hessian_evals = np.load(os.path.join(args.save_path, "hessian", f"hessian_evalue_{chosen_epoch}_adv_data.npy"))

        max_evals_1 = confidence * (single_gradient_norm ** 2)
        # max_evals_2 = single_hessian_evals[:, 0]

        max_evals_3 = confidence * (gradient_norm ** 2)
        # max_evals_4 = hessian_evals[:, 0]
        single_layer_evals.append(max_evals_1)
        all_layer_evals.append(max_evals_3)
        single_sample_lambda_hist(args, chosen_epoch, single_layer_evals[-1], "single_gradient")
        single_sample_lambda_hist(args, chosen_epoch, all_layer_evals[-1], "gradient")
        # compare_single_sample_lambda_max(args, chosen_epoch, max_evals_1, max_evals_2, "single_hessian")
        # compare_single_sample_lambda_max(args, chosen_epoch, max_evals_3, max_evals_4, "hessian")

        sevector = torch.load(os.path.join(args.model_path, "sevector", f"sevector_{chosen_epoch}.pkl"))
        max_sevector = sevector[:, 0].cpu().numpy()

        cos_sim = np.dot(max_sevector, single_gradient.transpose()) / (np.linalg.norm(max_sevector) * np.linalg.norm(single_gradient, axis=1))
        max_evals = np.sum(max_evals_1 * cos_sim)
        print()
    single_layer_evals = np.array(single_layer_evals)
    layer_evals = np.array(all_layer_evals)
    whole_evals = pd.read_csv(os.path.join(args.model_path, "hessian.csv"), sep=",", usecols=[6])
    whole_evals = whole_evals.values
    whole_evals = [float(re.findall("\d+.\d+", s)[0]) for s in whole_evals[:, 0]]

    last_single_layer_evals = single_layer_evals[-1]
    idx = np.argsort(last_single_layer_evals)
    used_interval = len(idx) // num
    used_idx = [idx[-1 - used_interval * i] for i in range(num)]
    compare_different_samples(args, single_layer_evals, used_idx, "single_gradient")

    # last_layer_evals = layer_evals[-1]
    # idx = np.argsort(last_layer_evals)
    # used_interval = len(idx) // num
    # used_idx = [idx[-1 - used_interval * i] for i in range(num)]
    compare_different_samples(args, layer_evals, used_idx, "gradient")
    print()


def cos_similarity(arr1, arr2):
    return np.sum(arr1 * arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))


def rank1_matrix_test():
    print("Test!")
    np.random.seed(0)
    a1 = np.random.randn(2, 1)
    a2 = np.random.randn(2, 1)
    mat1 = np.matmul(a1, a1.transpose())
    mat2 = np.matmul(a2, a2.transpose())
    c1 = np.linalg.eig(mat1)
    c2 = np.linalg.eig(mat2)
    c3 = np.linalg.eig(mat1 + mat2)

    cos1 = cos_similarity(c1[1][:, 0], c3[1][:, 1])
    cos2 = cos_similarity(c2[1][:, 1], c3[1][:, 1])

    eval = cos1 * c1[0].sum() + cos2 * c2[0].sum()
    print(a1, a2)
    print(c1, c2, c3)


if __name__ == '__main__':
    rank1_matrix_test()
