import torch
import torch.nn as nn
import numpy as np
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from clip import clip
from .element_wise_layer import Element_Wise_Layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIP_SD(nn.Module):
    def __init__(self, args, classnames,
                image_feature_dim=2048, num_classes=80, 
                word_feature_dim=512):
        super(CLIP_SD, self).__init__()
        self.clip_model = load_clip_to_cpu(args).float()
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.dtype = self.clip_model.dtype
        self.text_features = self.get_text_features(classnames)
        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        # self.classifiers = Element_Wise_Layer(1, self.image_feature_dim)
        self.fc1 = nn.Linear(self.image_feature_dim, self.image_feature_dim)
        self.fc2 = nn.Linear(self.image_feature_dim, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, image):
        image_features = self.clip_model.encode_image(image.type(self.dtype))       #[bs, 2048, H, W]
        text_features = self.text_features
        # SD
        sd_features = self.word_semantic(image_features, text_features)    # [bs, 80, 512]

        output = self.fc1(sd_features)
        output = self.fc2(self.relu(output))
        
        return output.squeeze()

    def get_text_features(self, classnames):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.clip_model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)
