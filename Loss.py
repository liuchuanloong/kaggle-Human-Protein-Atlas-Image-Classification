import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
class FocalLoss1(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.9,gamma=2):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,D].
        Return:
          (tensor) focal loss.
        '''
        # t = Variable(y).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*y + (1-p)*(1-y)         # pt = p if t > 0 else 1-p
        w = self.alpha*y + (1-self.alpha)*(1-y)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(self.gamma)
        t = torch.FloatTensor([  1.,   3.71186441,  93.13559322,   3.62711864,
        33.66101695,   9.05084746,  42.84745763,   1.94915254,
        16.83050847,  10.18644068,   2.37288136,   6.69491525,
         1.89830508,  12.03389831,   6.06779661,   7.76271186,
        12.3559322 ,   2.52542373,   4.06779661,   4.79661017,
        31.96610169,  12.25423729,  29.3559322 ,  21.06779661,
        17.94915254,  42.6440678 ,  14.16949153, 147.49152542])


        return F.binary_cross_entropy_with_logits(x, y, w, size_average=False,pos_weight=to_var(t))

# F.multilabel_margin_loss()
# nn.BCELoss()

if __name__ == '__main__':
    input = torch.FloatTensor([[0.7,0.3,-0.4,0.5,0.6],[0.3,-0.4,0.6,0.7,0.5]])
    label = torch.LongTensor([[1,0,0,0,1],[0,0,1,1,0]])
    loss = FocalLoss2()
    output = loss(input,label)
    print(output)
