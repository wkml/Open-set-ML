import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class semantic(nn.Module):
    def __init__(self, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self,image_features, word_features, num_classes=80):
        b, c, h, w = image_features.shape
        image_features = image_features.reshape(b, c, h * w).permute(2, 0, 1)   # (h * w, b, c)
        class_image_features = image_features.reshape(h * w, b, 1, c).repeat(1, 1, num_classes, 1)  # (h * w, b, num_classes, c)

        f_wh_feature = self.fc_1(image_features).reshape(h * w, b, 1, -1).repeat(1, 1, num_classes, 1)
        f_wd_feature = self.fc_2(word_features)
        lb_feature = self.fc_3(torch.tanh(f_wh_feature * f_wd_feature))
        coefficient = self.fc_a(lb_feature).reshape(h * w, b, num_classes)
        coefficient = F.softmax(coefficient, dim=0).permute(1, 2, 0)            # (b, num_classes, h * w)
        class_image_features = class_image_features.permute(3, 1, 2, 0)         # (c, b, num_classes, h * w)
        class_image_features = class_image_features * coefficient               # (c, b, num_classes, h * w)
        sd_features = torch.sum(class_image_features, dim=3).permute(1, 2, 0)   # (b, num_classes, c)
        return sd_features


