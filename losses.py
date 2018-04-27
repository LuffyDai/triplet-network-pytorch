import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLossSoftmax(nn.Module):

    def __init__(self, margin):
        super(TripletLossSoftmax, self).__init__()
        self.margin = margin

    def forward(self, dista, distb, size_average=True):
        distance_sum = dista.exp() + distb.exp()
        distance_positive = torch.div(distb.exp(), distance_sum)
        distance_negative = torch.div(dista.exp(), distance_sum) - self.margin
        losses = F.pairwise_distance(distance_positive, distance_negative, 2)
        return losses.mean() if size_average else losses.sum()


