import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label

    cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class=15, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        """
          input is B*N*C,
          target is B*N
        """
        # print(logit.shape)
        logit = F.softmax(input, dim=2)
        # print(logit.shape)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 1, 2).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        select=[]
        for i in range(target.shape[0]):
            if target[i,0]!=-100:
                select.append(i)

        select=torch.tensor(select)
        if select.device != logit.device:
            select = select.to(logit.device)
            
        logit=logit.index_select(dim=0,index=select)
        target=target.index_select(dim=0,index=select)


        idx = target
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(
            target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
        
        # print(one_hot_key,one_hot_key.shape)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        temp=one_hot_key * logit
        # print(temp)

        pt = (temp).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focalloss=MultiFocalLoss()
    def compute_loss(self,model,inputs,return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # print(logits)
        # print(logits.shape)
        # print(labels)
        # print(labels.shape)
        loss = self.focalloss(logits,labels)
        return (loss,outputs) if return_outputs else loss 

if __name__ == "__main__":
    Predict = torch.tensor([[[0, 0, 1000000], [1000000, 0, 0]],[[0, 0, 1000000], [1000000, 0, 0]]]).float()
    Target = torch.tensor([[2, -100],[-100, 0]]).float()
    mul = MultiFocalLoss(3)
    print(mul(Predict, Target))
