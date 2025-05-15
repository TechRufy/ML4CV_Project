import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class CrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0, beta=0, gamma=0, size_average=True, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, logit, target, features_in):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, size_average=self.size_average)

        if self.cuda:
            criterion = criterion.cuda()

        CE_loss = criterion(logit, target.squeeze(1))
        VAR_loss = Variable(torch.Tensor([0]))
        Inter_loss = Variable(torch.Tensor([0]))
        Center_loss = Variable(torch.Tensor([0]))
        for i in range(n):
            label = target[i]
            label = label.flatten().cpu().numpy()
            features = logit[i]
            features = features.permute(1, 2, 0).contiguous()
            shape = features.size()
            features = features.view(shape[0] * shape[1], shape[2])
            features_in_temp = features_in[i]

            instances, counts = np.unique(label, False, False, True)
            # print('counts', counts)
            total_size = int(np.sum(counts))
            for instance in instances:

                if instance == self.ignore_index:  # Ignore background
                    continue

                locations = torch.LongTensor(np.where(label == instance)[0]).cuda()
                vectors = torch.index_select(features, dim=0, index=locations)
                features_temp = torch.index_select(features_in_temp, dim=0, index=locations)
                centers_temp = torch.mean(features_temp, dim=0)
                features_temp = features_temp - centers_temp
                Center_loss += torch.sum(features_temp ** 2) / total_size
                # print(size)
                # print(-vectors[:,int(instance)])
                # get instance mean and distances to mean of all points in an instance
                VAR_loss += torch.sum((-vectors[:, int(instance)])) / total_size
                Inter_loss += (torch.sum(vectors) - torch.sum((vectors[:, int(instance)]))) / total_size

                # total_size += size

            # VAR_loss += var_loss/total_size

        loss = (CE_loss + self.alpha * VAR_loss + self.beta * Inter_loss + self.gamma * Center_loss) / n
        # print(CE_loss/n, self.alpha * VAR_loss/n, self.beta * Inter_loss/n, self.gamma * Center_loss/n)

        return loss
