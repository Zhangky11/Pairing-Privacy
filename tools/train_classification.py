import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data

# import internal libs
import model
from tools.plot import plot_curves
from tools.hessian import lanczos, get_hessian, get_hessian_single_layer
from tools.utils import makedirs
from tools.adversarial import adversarial_dataset, save_adv_samples
from torch.nn.utils import parameters_to_vector
from tools.evaluate_single_sample import evaluate_single_sample
from tools.evaluate import evaluate


def train_classification_model(args, X_train, y_train, X_test, y_test, train_loader):
    """prepare the model the later usage. If the model haven't been trained, we will firstly
    train the model. If the model has been trained, we will certainly load the model.

    Args:
        args (Namespace): set containing all program arguments.
        X_train, y_train, X_test, y_test (tensor.to(deivice)): splitted dataset.
        train_loader (nn.dataloader) used for training model

    Returns:
        return the fitted model or trained model.
    """
    # get the net
    in_dims = X_train.size(-1)
    out_dims = len(set(y_train.detach().cpu().numpy()))
    if args.dataset in ["census", "commercial"] \
            + [f"gaussian_rule{i}_classification_ind" for i in range(1, 11)] \
            + [f"gaussian_8d_rule{i}_classification" for i in range(1, 11)]:
        net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=out_dims)
    else:
        raise Exception(f"Unknown dataset: {args.dataset} for multi-classification.")
    print(net)
    net = net.float().to(args.device)

    # define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)

    if "model.pt" in os.listdir(args.model_path):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.model_path))
        net.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt")))

        # evaluate the performance of the model
        net = eval_classification_model(net, X_test, y_test, criterion)

        return net
    else:
        print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.model_path))
        if args.model_seed != args.seed:
            print("argument model_seed should be as same as argument seed to train a new model with new settings. Terminate program with code -3.")
            exit(-3)

        # define the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.train_lr)
        # set the decay of learning rate
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.train_lr), np.log10(args.train_lr) - args.logspace, args.epoch)

        # define the train_csv
        learning_csv = os.path.join(args.model_path, "learning.csv")
        # define the res dict
        res_dict = {
            'train-loss': [], 'train-acc': [], 'test-loss': [], "test-acc": []
        }

        # starts training
        for epoch in range(args.epoch + 1):  # to test the acc for the last time
            # eval on test set
            net.eval()
            with torch.no_grad():
                # eval on train
                train_log_probs = net(X_train)
                train_loss = criterion(train_log_probs, y_train)
                train_targets = torch.argmax(train_log_probs, dim=-1)
                train_acc = torch.sum(train_targets == y_train).item() / len(y_train)

                # eval on test
                test_log_probs = net(X_test)
                test_loss = criterion(test_log_probs, y_test)
                test_targets = torch.argmax(test_log_probs, dim=-1)
                test_acc = torch.sum(test_targets == y_test).item() / len(y_test)

            # save the res in dict
            res_dict['train-loss'].append(train_loss.item())
            res_dict["train-acc"].append(train_acc)
            res_dict['test-loss'].append(test_loss.item())
            res_dict["test-acc"].append(test_acc)
            # store the res in csv
            pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

            # show loss
            if epoch % 10 == 0 or epoch == args.epoch:
                print('On train set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, train_loss.item(), train_acc))
                print('On test set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, test_loss.item(), test_acc))

                if epoch % 100 == 0 or epoch == args.epoch - 1:
                    # draw the curves
                    plot_curves(args.model_path, res_dict)

            if epoch >= args.epoch:
                break

            # set the lr
            if args.logspace != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]
            for step, (batch_X, batch_y) in enumerate(train_loader):
                # print(batch_y.sum() / len(batch_y))
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                net.train()
                net.zero_grad()
                log_probs = net(batch_X)
                loss = criterion(log_probs, batch_y)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                loss.backward()
                optimizer.step()

        plot_curves(args, args.model_path, res_dict)
        # save the model
        torch.save(net.cpu().state_dict(), os.path.join(args.model_path, "model_final.pt"))
        print("The model has been trained and saved in model path '{}'.".format(args.model_path))

        return net


def eval_classification_model(net, X_test, y_test, criterion):
    # evaluate the performance of the model
    net.eval()
    with torch.no_grad():
        logits = net(X_test)
        loss = criterion(logits, y_test)
        targets = torch.argmax(logits, dim=-1)
        acc = torch.sum(targets == y_test).item() / len(y_test)

    print('On test set - \t Loss: {:.6f} \t Acc: {:.9f}'.format(loss.item(), acc))

    return net


def train_classification_model_hessian(args, X_train, y_train, X_test, y_test, train_loader):
    """prepare the model the later usage. If the model haven't been trained, we will firstly
    train the model. If the model has been trained, we will certainly load the model.

    Args:
        args (Namespace): set containing all program arguments.
        X_train, y_train, X_test, y_test (tensor.to(deivice)): splitted dataset.
        train_loader (nn.dataloader) used for training model

    Returns:
        return the fitted model or trained model.
    """
    # get the net
    in_dims = X_train.size(-1)
    out_dims = len(set(y_train.detach().cpu().numpy()))
    if args.dataset in ["census", "commercial"] \
            + [f"gaussian_rule{i}_classification_ind" for i in range(1, 11)] \
            + [f"gaussian_8d_rule{i}_classification" for i in range(1, 11)]:
        net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=out_dims)
    else:
        raise Exception(f"Unknown dataset: {args.dataset} for multi-classification.")
    print(net)
    net = net.float().to(args.device)

    # define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)

    train_set = Data.TensorDataset(X_train, y_train)

    if "model_final.pt" in os.listdir(args.model_path):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.model_path))
        net.load_state_dict(torch.load(os.path.join(args.model_path, "model_final.pt")))

        # evaluate the performance of the model
        net = eval_classification_model(net, X_test, y_test, criterion)

        # evaluate hessian information of the model during the training process
        evaluate(args, net, X_train, y_train, criterion)
        return net
    else:
        print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.model_path))
        if args.model_seed != args.seed:
            print("argument model_seed should be as same as argument seed to train a new model with new settings. Terminate program with code -3.")
            exit(-3)

        # define the optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=args.train_lr)
        # set the decay of learning rate
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.train_lr), np.log10(args.train_lr) - args.logspace, args.epoch)

        # define the train_csv
        learning_csv = os.path.join(args.model_path, "learning.csv")
        hessian_csv = os.path.join(args.model_path, "hessian.csv")
        # define the res dict
        res_dict = {
            'train-loss': [], 'train-acc': [], 'test-loss': [], "test-acc": []
        }
        hessian_dict = {
            'eigen_1': [], 'eigen_2': [], 'eigen_3': [], 'eigen_4': [], 'eigen_5': [], 'eigen_6': [],
            's_eigen_1': [], 's_eigen_2': [], 's_eigen_3': [], 's_eigen_4': [], 's_eigen_5': [], 's_eigen_6': []
        }

        # starts training
        for epoch in range(args.epoch + 1):  # to test the acc for the last time
            # eval on test set
            net.eval()
            with torch.no_grad():
                # eval on train
                train_log_probs = net(X_train)
                train_loss = criterion(train_log_probs, y_train)
                train_targets = torch.argmax(train_log_probs, dim=-1)
                train_acc = torch.sum(train_targets == y_train).item() / len(y_train)

                # eval on test
                test_log_probs = net(X_test)
                test_loss = criterion(test_log_probs, y_test)
                test_targets = torch.argmax(test_log_probs, dim=-1)
                test_acc = torch.sum(test_targets == y_test).item() / len(y_test)

            # ======== Get Hessian Eigenvalues ====================================
            if epoch % args.hessian_interval == 0 or epoch == args.epoch:
                p = len(parameters_to_vector(net.parameters()))
                sp = len(parameters_to_vector(net.layers[0].parameters()))
                criterion_hessian = nn.CrossEntropyLoss(reduction="sum")
                lam_get_hessian = lambda delta: get_hessian(args, net, criterion_hessian, train_loader, delta)
                lam_get_hessian_single_layer = lambda delta: get_hessian_single_layer(args, net, net.layers[0],
                                                                                      criterion_hessian,
                                                                                      train_loader, delta)
                evals, evecs = lanczos(lam_get_hessian, p, neigs=6)
                sevals, sevecs = lanczos(lam_get_hessian_single_layer, sp, neigs=6)
                hessian_dict['eigen_1'].append(evals[0])
                hessian_dict['eigen_2'].append(evals[1])
                hessian_dict['eigen_3'].append(evals[2])
                hessian_dict['eigen_4'].append(evals[3])
                hessian_dict['eigen_5'].append(evals[4])
                hessian_dict['eigen_6'].append(evals[5])

                hessian_dict['s_eigen_1'].append(sevals[0])
                hessian_dict['s_eigen_2'].append(sevals[1])
                hessian_dict['s_eigen_3'].append(sevals[2])
                hessian_dict['s_eigen_4'].append(sevals[3])
                hessian_dict['s_eigen_5'].append(sevals[4])
                hessian_dict['s_eigen_6'].append(sevals[5])
            # =====================================================================

            # save the res in dict
            res_dict['train-loss'].append(train_loss.item())
            res_dict["train-acc"].append(train_acc)
            res_dict['test-loss'].append(test_loss.item())
            res_dict["test-acc"].append(test_acc)
            # store the res in csv
            pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)
            pd.DataFrame.from_dict(hessian_dict).to_csv(hessian_csv, index=False)

            # show loss
            if epoch % 10 == 0 or epoch == args.epoch:
                print('On train set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, train_loss.item(), train_acc))
                print('On test set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, test_loss.item(), test_acc))
                print('On test set - Epoch: {} \t Eigen values: {}'
                      .format(epoch, evals))
                print('On test set - Epoch: {} \t Single layer eigen values: {}'
                      .format(epoch, sevals))

                evaluate_single_sample(args, train_set, net, criterion, mode="gradient",
                                       filename=f"{epoch}_adv_data")
                evaluate_single_sample(args, train_set, net, criterion, mode="confidence",
                                       filename=f"{epoch}_adv_data")
                evaluate_single_sample(args, train_set, net, criterion, mode="single_gradient",
                                       filename=f"{epoch}_adv_data")

                if epoch % 100 == 0 or epoch == args.epoch - 1:
                    # draw the curves
                    plot_curves(args, args.model_path, res_dict)
                    plot_curves(args, args.model_path, hessian_dict)
                    # save the model
                    torch.save(net.cpu().state_dict(), os.path.join(args.model_path, f"model_{epoch}.pt"))
                    net.to(args.device)

            if epoch >= args.epoch:
                break

            # set the lr
            if args.logspace != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]
            for step, (batch_X, batch_y) in enumerate(train_loader):
                # print(batch_y.sum() / len(batch_y))
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                net.train()
                net.zero_grad()
                log_probs = net(batch_X)
                loss = criterion(log_probs, batch_y)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                loss.backward()
                optimizer.step()

        plot_curves(args, args.model_path, res_dict)
        # save the model
        torch.save(net.cpu().state_dict(), os.path.join(args.model_path, "model_final.pt"))
        print("The model has been trained and saved in model path '{}'.".format(args.model_path))

        return net


def adv_train_classification_model_hessian(args, X_train, y_train, X_test, y_test, train_loader):
    """prepare the model the later usage. If the model haven't been trained, we will firstly
    train the model. If the model has been trained, we will certainly load the model.

    Args:
        args (Namespace): set containing all program arguments.
        X_train, y_train, X_test, y_test (tensor.to(deivice)): splitted dataset.
        train_loader (nn.dataloader) used for training model

    Returns:
        return the fitted model or trained model.
    """
    # get the net
    in_dims = X_train.size(-1)
    out_dims = len(set(y_train.detach().cpu().numpy()))
    if args.dataset in ["census", "commercial"] \
            + [f"gaussian_rule{i}_classification_ind" for i in range(1, 11)] \
            + [f"gaussian_8d_rule{i}_classification" for i in range(1, 11)]:
        net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=out_dims)
    else:
        raise Exception(f"Unknown dataset: {args.dataset} for multi-classification.")
    print(net)
    net = net.float().to(args.device)

    # define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)

    if "model_final.pt" in os.listdir(args.model_path):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.model_path))
        net.load_state_dict(torch.load(os.path.join(args.model_path, "model_final.pt")))

        # evaluate the performance of the model
        net = eval_classification_model(net, X_test, y_test, criterion)

        # evaluate hessian information of the model during the training process
        evaluate(args, net, X_train, y_train, criterion)

        # evaluate the adversarial process
        # adv_train_set, adv_train_loader, adv_losses = adversarial_dataset(args, net, X_train.clone(), y_train.clone(), criterion)

        return net
    else:
        print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.model_path))
        if args.model_seed != args.seed:
            print("argument model_seed should be as same as argument seed to train a new model with new settings. Terminate program with code -3.")
            exit(-3)

        # define the optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=args.train_lr)
        # set the decay of learning rate
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.train_lr), np.log10(args.train_lr) - args.logspace, args.epoch)

        # define the train_csv
        learning_csv = os.path.join(args.model_path, "learning.csv")
        hessian_csv = os.path.join(args.model_path, "hessian.csv")
        # define the res dict
        res_dict = {
            'train-loss': [], 'train-acc': [], 'test-loss': [], "test-acc": []
        }
        hessian_dict = {
            'eigen_1': [], 'eigen_2': [], 'eigen_3': [], 'eigen_4': [], 'eigen_5': [], 'eigen_6': [],
            'ori_eigen_1': [], 'ori_eigen_2': [], 'ori_eigen_3': [], 'ori_eigen_4': [], 'ori_eigen_5': [], 'ori_eigen_6': [],
            's_eigen_1': [], 's_eigen_2': [], 's_eigen_3': [], 's_eigen_4': [], 's_eigen_5': [], 's_eigen_6': []
        }

        # starts training
        for epoch in range(args.epoch + 1):  # to test the acc for the last time
            # Generate adv train loader
            adv_train_set, adv_train_loader, adv_losses = adversarial_dataset(args, net, X_train.clone(), y_train.clone(), criterion)
            # eval on test set
            net.eval()
            with torch.no_grad():
                # eval on train
                train_log_probs = net(X_train)
                train_loss = criterion(train_log_probs, y_train)
                train_targets = torch.argmax(train_log_probs, dim=-1)
                train_acc = torch.sum(train_targets == y_train).item() / len(y_train)

                # eval on test
                test_log_probs = net(X_test)
                test_loss = criterion(test_log_probs, y_test)
                test_targets = torch.argmax(test_log_probs, dim=-1)
                test_acc = torch.sum(test_targets == y_test).item() / len(y_test)

            # ======== Get Hessian Eigenvalues ====================================
            if epoch % args.hessian_interval == 0 or epoch == args.epoch:
                p = len(parameters_to_vector(net.parameters()))
                sp = len(parameters_to_vector(net.layers[0].parameters()))
                criterion_hessian = nn.CrossEntropyLoss(reduction="sum")
                lam_get_hessian = lambda delta: get_hessian(args, net, criterion_hessian, adv_train_loader, delta)
                lam_get_ori_hessian = lambda delta: get_hessian(args, net, criterion_hessian, train_loader, delta)
                lam_get_hessian_single_layer = lambda delta: get_hessian_single_layer(args, net, net.layers[0], criterion_hessian, adv_train_loader, delta)
                evals, evecs = lanczos(lam_get_hessian, p, neigs=6)
                sevals, sevecs = lanczos(lam_get_hessian_single_layer, sp, neigs=6)
                ori_evals, ori_evecs = lanczos(lam_get_ori_hessian, p, neigs=6)
                hessian_dict['eigen_1'].append(evals[0])
                hessian_dict['eigen_2'].append(evals[1])
                hessian_dict['eigen_3'].append(evals[2])
                hessian_dict['eigen_4'].append(evals[3])
                hessian_dict['eigen_5'].append(evals[4])
                hessian_dict['eigen_6'].append(evals[5])

                hessian_dict['s_eigen_1'].append(sevals[0])
                hessian_dict['s_eigen_2'].append(sevals[1])
                hessian_dict['s_eigen_3'].append(sevals[2])
                hessian_dict['s_eigen_4'].append(sevals[3])
                hessian_dict['s_eigen_5'].append(sevals[4])
                hessian_dict['s_eigen_6'].append(sevals[5])

                hessian_dict['ori_eigen_1'].append(ori_evals[0])
                hessian_dict['ori_eigen_2'].append(ori_evals[1])
                hessian_dict['ori_eigen_3'].append(ori_evals[2])
                hessian_dict['ori_eigen_4'].append(ori_evals[3])
                hessian_dict['ori_eigen_5'].append(ori_evals[4])
                hessian_dict['ori_eigen_6'].append(ori_evals[5])
            # =====================================================================

            # save the res in dict
            res_dict['train-loss'].append(train_loss.item())
            res_dict["train-acc"].append(train_acc)
            res_dict['test-loss'].append(test_loss.item())
            res_dict["test-acc"].append(test_acc)
            # store the res in csv
            pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)
            pd.DataFrame.from_dict(hessian_dict).to_csv(hessian_csv, index=False)

            # show loss
            if epoch % 10 == 0 or epoch == args.epoch:
                print('On train set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, train_loss.item(), train_acc))
                print('On test set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                      .format(epoch, test_loss.item(), test_acc))
                print('On train set - Epoch: {} \t Eigen values: {}'
                      .format(epoch, evals))
                print('On train set - Epoch: {} \t Ori eigen values: {}'
                      .format(epoch, ori_evals))
                print('On train set - Epoch: {} \t Single layer eigen values: {}'
                      .format(epoch, sevals))

                # evaluate_single_sample(args, adv_train_set, net, criterion, mode="gradient",
                #                        filename=f"{epoch}_adv_data")
                # evaluate_single_sample(args, adv_train_set, net, criterion, mode="confidence",
                #                        filename=f"{epoch}_adv_data")
                # evaluate_single_sample(args, adv_train_set, net, criterion, mode="single_gradient",
                #                        filename=f"{epoch}_adv_data")

                if epoch % 50 == 0 or epoch == args.epoch - 1:
                    # draw the curves
                    plot_curves(args, args.model_path, res_dict)
                    plot_curves(args, args.model_path, hessian_dict)

                    # save adv samples
                    save_adv_samples(args, epoch, adv_losses, train_loader, adv_train_loader)
                    makedirs(os.path.join(args.model_path, "adv_dataset"))
                    torch.save(adv_train_loader.dataset.tensors, os.path.join(args.model_path, "adv_dataset", f"adv_dataset_{epoch}.pt"))
                    makedirs(os.path.join(args.model_path, "evector"))
                    torch.save(evecs, os.path.join(args.model_path, "evector", f"evector_{epoch}.pkl"))
                    makedirs(os.path.join(args.model_path, "ori_evector"))
                    torch.save(ori_evecs, os.path.join(args.model_path, "ori_evector", f"ori_evector_{epoch}.pkl"))
                    makedirs(os.path.join(args.model_path, "sevector"))
                    torch.save(sevecs, os.path.join(args.model_path, "sevector", f"sevector_{epoch}.pkl"))
                    # evaluate single sample
                    # evaluate_single_sample(args, adv_train_set, net, criterion, mode="single_hessian", filename=f"{epoch}_adv_data")
                    evaluate_single_sample(args, adv_train_set, net, criterion, mode="single_gradient", filename=f"{epoch}_adv_data")
                    # evaluate_single_sample(args, adv_train_set, net, criterion, mode="hessian", filename=f"{epoch}_adv_data")
                    evaluate_single_sample(args, adv_train_set, net, criterion, mode="gradient", filename=f"{epoch}_adv_data")
                    evaluate_single_sample(args, adv_train_set, net, criterion, mode="confidence", filename=f"{epoch}_adv_data")
                    # save the model
                    makedirs(os.path.join(args.model_path, "model"))
                    torch.save(net.cpu().state_dict(), os.path.join(args.model_path, "model", f"model_{epoch}.pt"))
                    net.to(args.device)

            if epoch >= args.epoch:
                break

            # set the lr
            if args.logspace != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]

            adv_train_loader = Data.DataLoader(adv_train_set, batch_size=args.batch_size, shuffle=True)
            for step, (batch_X, batch_y) in enumerate(adv_train_loader):
                # print(batch_y.sum() / len(batch_y))
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                net.train()
                net.zero_grad()
                log_probs = net(batch_X)
                if args.norm_grad:
                    def norm_grad(grad):
                        grad = grad / grad.norm(dim=1).view(-1, 1)
                        if step == 0:
                            print(grad)
                            print(grad.norm(dim=1))
                        # print(grad)
                        return grad
                    hook = log_probs.register_hook(norm_grad)
                loss = criterion(log_probs, batch_y)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                loss.backward()
                optimizer.step()
                if args.norm_grad:
                    hook.remove()

        plot_curves(args, args.model_path, res_dict)
        plot_curves(args, args.model_path, hessian_dict)
        # save the model
        torch.save(net.cpu().state_dict(), os.path.join(args.model_path, "model_final.pt"))
        print("The model has been trained and saved in model path '{}'.".format(args.model_path))

        return net
