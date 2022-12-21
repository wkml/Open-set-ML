import torch
import torch.nn as nn
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from clip import clip
from .element_wise_layer import Element_Wise_Layer

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
        self.word_semantic = semantic(num_classes=80,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim,
                                      intermediary_dim=256)

        # self.classifiers = Element_Wise_Layer(1, self.image_feature_dim)
        self.logit_scale = self.clip_model.logit_scale
        # self.fc = nn.Linear(self.image_feature_dim, 512)
        self.relu = nn.ReLU()

    def forward(self, image, train=True):
        image_features, _ = self.clip_model.encode_image(image.type(self.dtype))       #[bs, 2048, H, W]
        
        # SD
        if train:
            text_features = self.text_features[:60]
            sd_features = self.word_semantic(image_features, text_features, 60)    # [bs, 80, 512]
        else:
            text_features = self.text_features
            sd_features = self.word_semantic(image_features, text_features, 80)

        sd_features = self.relu(sd_features)
        
        sd_features = sd_features / sd_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sd_features @ text_features.t()          # [bs, 80, 80]

        output = torch.diagonal(logits, dim1=-2, dim2=-1)
        
        return output

    def get_text_features(self, classnames):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.clip_model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)
    