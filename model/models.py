import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from clip import clip
from .element_wise_layer import Element_Wise_Layer

class SSCLIP(nn.Module):
    def __init__(self, args, word_features, classname,
                image_feature_dim=512, num_classes=80, 
                word_feature_dim=300):
        super(SSCLIP, self).__init__()
        self.clip_model = load_clip_to_cpu(args).float()
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        self.token = clip.tokenize(classname).cpu()
        self.text_features = self.get_text_features(classname)

        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)
        self.word_features = self.load_features(word_features)

        # self.fc_output = nn.Linear(2*self.image_feature_dim, self.output_dim)
        self.classifiers = Element_Wise_Layer(self.num_classes, self.image_feature_dim)
        # self.fc2 = nn.Linear(self.num_classes, 1)
        # self.fc = nn.Linear(80 * self.image_feature_dim, 512)

        self.temperature = nn.Parameter(torch.tensor(3.0, dtype=self.dtype))

        self.v_linear_weight = self.clip_model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.clip_model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.clip_model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.clip_model.visual.attnpool.c_proj.bias



    def forward(self, image):
        # with attn
        # img_feature_map, _ = self.clip_model.encode_image(image.type(self.dtype))  #[bs, 512, H, W]

        # without attn
        x = self.clip_model.encode_image(image.type(self.dtype))       #[bs, 2048, H, W]
        x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
        x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
        img_feature_map = x

        # SD
        sd_features = self.word_semantic(img_feature_map, self.word_features)    # [bs, 80, 512]

        output = self.classifiers(torch.tanh(sd_features))

        # sd_features = sd_features.reshape(-1, 1, self.num_classes * self.image_feature_dim)     # [bs, 1, 80 * 2048]
        # sd_features = self.fc(sd_features)                      # [bs, 1, 512]
        text_features = self.text_features                        # [80, 512]

        sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)  
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ?
        logits_scale = self.temperature.exp()
        logits = logits_scale * sd_features @ text_features.t()   # [bs, 80, 80]
        
        return output, logits
    
    def load_features(self,word_features):
        return nn.Parameter(torch.from_numpy(np.load(word_features).astype(np.float32)), requires_grad=False)

    def get_text_features(self, classname):
        token = clip.tokenize(classname).cpu()
        return nn.Parameter(self.clip_model.encode_text(token), requires_grad=False)
    
    # def load_text_feature(self, text_features):
    #     return torch.from_numpy(np.load(text_features).astype(np.float32))