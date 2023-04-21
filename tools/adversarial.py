import torch.utils.data as Data
import torch
import os
import json


def save_adv_samples(args, epoch, adv_losses, train_loader, adv_train_loader):
    adv_samples = {}
    for i in range(args.visual_adv_num):
        adv_samples[str(i)] = {}
        adv_samples[str(i)]["x"] = train_loader.dataset.tensors[0][i].cpu().numpy().tolist()
        adv_samples[str(i)]["y"] = [train_loader.dataset.tensors[1][i].item()]
        adv_samples[str(i)]["adv_x"] = adv_train_loader.dataset.tensors[0][i].cpu().numpy().tolist()
        adv_samples[str(i)]["loss"] = adv_losses[i]
    with open(os.path.join(args.save_path, f"adv_samples_epoch-{epoch}.json"), "w") as file:
        json.dump(adv_samples, file, indent=4)



def adversarial_data(args, x):
    assert args.attack_mode in ["FGSM", "PGD", "Grad"]
    if args.attack_mode == "FGSM":
        x = x + x.grad.sign() * args.adv_lr
    elif args.attack_mode == "PGD":
        x = x + x.grad.sign() * args.adv_lr
    elif args.attack_mode == "Grad":
        x = x + x.grad * args.adv_lr
    return x


def adversarial_dataset(args, model, X_train, y_train, criterion):
    if args.attack_mode == "FGSM":
        args.adv_step = 1
    train_set = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(train_set, batch_size=30000, shuffle=False)

    losses = [[] for _ in range(args.visual_adv_num)]
    acc = 0
    adv_acc = 0
    for step, (batch_X, batch_y) in enumerate(train_loader):
        flag = False
        for adv_step in range(args.adv_step):
            batch_X = batch_X.detach().requires_grad_(True)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            if adv_step == 0:
                acc += batch_y[0] == output.argmax(1)[0]
                flag = True
            batch_X = adversarial_data(args, batch_X)

            # if step < args.visual_adv_num:
            if step == 0:
                with torch.no_grad():
                    for i in range(args.visual_adv_num):
                        x = batch_X[i].detach().unsqueeze(0)
                        y = model(x)
                        new_loss = criterion(y, batch_y[i].unsqueeze(0))
                        losses[i].append(new_loss.item())
        batch_X = batch_X.detach()
        output = model(batch_X)
        if flag:
            adv_acc += batch_y[0] == output.argmax(1)[0]

        start = step
        X_train[start: start + batch_X.size(0)] = batch_X.detach()
    adv_train_set = Data.TensorDataset(X_train, y_train)
    adv_train_loader = Data.DataLoader(adv_train_set, batch_size=args.batch_size, shuffle=False)
    return adv_train_set, adv_train_loader, losses
