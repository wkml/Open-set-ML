import torch
from torch import nn

class AsymmetricLoss(nn.Module):
    def __init__(self, margin=0, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce=None, size_average=None):
        super(AsymmetricLoss, self).__init__()

        self.reduce = reduce
        self.size_average = size_average

        self.margin = margin
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, input, target):
        """
        Shape of input: (BatchSize, classNum)
        Shape of target: (BatchSize, classNum)
        """

        # Get positive and negative mask
        positive_mask = (target > self.margin).float()
        negative_mask = (target < -self.margin).float()

        # Calculating Probabilities
        input_sigmoid = torch.sigmoid(input)
        input_sigmoid_pos = input_sigmoid
        input_sigmoid_neg = 1 - input_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            input_sigmoid_neg = (input_sigmoid_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = positive_mask * torch.log(input_sigmoid_pos.clamp(min=self.eps))
        loss_neg = negative_mask * torch.log(input_sigmoid_neg.clamp(min=self.eps))
        loss = -1 * (loss_pos + loss_neg)

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = input_sigmoid_pos * positive_mask + input_sigmoid_neg * negative_mask
            one_sided_gamma = self.gamma_pos * positive_mask + self.gamma_neg * negative_mask
            one_sided_weight = torch.pow(1 - prob, one_sided_gamma)

            loss *= one_sided_weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss