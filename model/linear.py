import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x

# ===========================
#   wrapper
# ===========================
def linear(in_dim, out_dim):
    return MLP(in_dim, out_dim)



if __name__ == '__main__':
    x = torch.rand(500,2)
    net = linear(in_dim=2, out_dim=1)
    print(net)
    print(net(x).shape)