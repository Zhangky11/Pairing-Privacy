import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))
sys.path.append("./model")
from read_tabular import read_tabular
# from net3_onlyW import PrunedNet
from model.net4 import PrunedNet
# from model.net3_W_V import draw_hist
from loss import criterion2

input_size = 12
output_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train, y_train, X_test, y_test, train_loader, test_loader = read_tabular()

# test
model = PrunedNet(input_size, output_size)
model.to(device)
path = './NetModel.pth'
model.load_state_dict(torch.load(path))

# for name, param in model.named_parameters():
#     if param.requires_grad and (name[0] == 'v' or name[0] == 'w'):
#         # torch.set_printoptions(profile="full")
#         print(name, "data", param.data)
#         print(name, "grad", param.grad)

# get position for classification "1" in batch
pos_to_be_tested = torch.zeros(1)
for i, (x_te, y_te) in enumerate(train_loader):
    if i > 0: break
    torch.set_printoptions(profile="full")
    
    y_te = y_te.to(device)
    x_te = x_te.to(device)
    y_pred = model(x_te, 0)  # (512, 2)
    y_classification = y_pred.max(1)[1]
    torch.set_printoptions(profile="default") # reset.
    # print(f"y_pred:{y_pred}")
    # print(f"y_classification:{y_classification}")

    #print(f"y_te:{y_te}")
    # find the position which satisfy: 1.real class is "1", 2. the prediction is also "1"
    # pos_to_be_tested = ((y_te & y_classification) == 1).nonzero(as_tuple=True)[0]

for name, param in model.named_parameters():
            if param.requires_grad :  # (name[0] == 'v') and 
                # torch.set_printoptions(profile="full")
                torch.set_printoptions(profile="full")
                print(name, "data", param.data)
                print(name, "grad", param.grad)
                torch.set_printoptions(profile="default") # reset.
                # sorted data and grad
                """
                sorted, _ = torch.sort(param.data)
                print("sorted data: ",name,sorted)
                sorted, _ = torch.sort(param.grad)
                print("sorted grad: ",name,sorted)
                """
                # torch.set_printoptions(profile="default")
torch. set_printoptions(profile="full")
print("model.p2: ", model.p2)
print("model.final_child for p2: ", model.final_child1)
print("model.p2.shape: ", model.p2.shape)
print("model.final_child for p3: ", model.final_child2)
print("model.p3: ", model.p3)
print("model.p3.shape: ", model.p3.shape)
torch. set_printoptions(profile="default") # reset.

# print("pos_to_be_tested: ", pos_to_be_tested)

# total_test_num = 6033

# print("total_test_num: ", total_test_num)

'''
# exit()
ones_after = []
for t in pos_to_be_tested[:]:  #全为1的样本
    print(f"t:{t}")
    x_replace = torch.zeros(1)
    zero_count_before = 0
    one_count_before = 0
    zero_count_after = 0
    one_count_after = 0
    cnt_not_in_after_one_pos = 0
    print(f"\n the position to be changed is the {t}th input in batch 1\n") #循环第i个分类为1 label为1的样本
    for i, (x_te, y_te) in enumerate(test_loader):
        # print(i)
        model.eval()
        # print(i)
        # if i > 0:
        #     break
        with torch.no_grad():
            x_te = x_te.to(device)
            y_te = y_te.to(device)
            batchsize = len(x_te)  # 512
            # print("before change, x_te: ", x_te)
            # print("y_te: ", y_te)
            y_pred = model(x_te, 0)  # (512, 2)
            y_classification = y_pred.max(1)[1]
            #print(f"y_classification:{y_classification}")
            before_one_pos = y_classification.nonzero(as_tuple=True)[0]
            #print(f"before_one_pos:{before_one_pos}, sum:{y_classification.sum()}")

            one_count_before += y_classification.sum()
            zero_count_before += batchsize - y_classification.sum()

            # correct_y = y_pred.max(1)[1] == y_te
            # print("before change, y_classification: ", y_classification)

            # the postion to change input
            pos = [2, 3, 5, 7, 8, 10, 11] #important pos
            # pos = [1, 3, 4, 5, 6] #not important pos
            # the input to be changed
            change_input = [i for i in range(batchsize)]

            # define x_replace as the the "t"th input in the first batch
            if i == 0:
                #print(f"t:{t}")
                print(x_te[t, :])
                x_replace = x_te[t, :]
                # print(x_te[t, :])

            # change the important dimension (need be specified w.r.t. model)
            # of all other input to the same as x_repalce
            for k in pos:
                for j in change_input:
                    x_te[j, k] = x_replace[k]

            # print("after change, x_te: ", x_te)
            y_pred = model(x_te, 0)  # (512, 2)
            y_classification = y_pred.max(1)[1]
            after_one_pos = y_classification.nonzero(as_tuple=True)[0]
            #print(f"after_one_pos:{after_one_pos}, sum:{y_classification.sum()}")
            cnt = 0
            for p in before_one_pos:
                if p not in after_one_pos:
                    #print(f"{p}-th pos is not in after_one_pos")
                    cnt = cnt + 1
            cnt_not_in_after_one_pos += cnt
            #print(f"pos is not in after_one_pos:{cnt}")

            # correct_y = y_pred.max(1)[1] == y_te
            # print("after change, y_classification: ", y_classification)
            one_count_after += y_classification.sum()
            zero_count_after += batchsize - y_classification.sum()
            #if i == 11:

    ones_after.append(one_count_after.cpu().detach().item())
    print("cnt_not_in_after_one_pos", cnt_not_in_after_one_pos)
    print("one_count_before: ", one_count_before)
    print("one_reserved:", one_count_before-cnt_not_in_after_one_pos)
    # print("zero_count_before: ", zero_count_before)
    print("one_count_after: ", one_count_after)
    print("zero_to_one:", one_count_after-(one_count_before-cnt_not_in_after_one_pos))
    # print("zero_count_after ", zero_count_after)

            # print("y_pred.max(1)[1]: ", y_pred.max(1)[1])
            # test_correct += (y_pred.max(1)[1] == y_te).sum().item()

print("ones_after: ", ones_after)
'''
