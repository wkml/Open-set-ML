import torch
import torch.nn as nn
from model.prompt_learner import PromptLearner, TextEncoder
from utils.checkpoint import load_clip_to_cpu
from clip import clip
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIP_SD(nn.Module):
    def __init__(self, args, classnames):
        super(CLIP_SD, self).__init__()
        self.num_classes = len(classnames)
        self.clip_model = load_clip_to_cpu(args).float()
        self.dtype = self.clip_model.dtype
        self.prompt_learner = PromptLearner(args, classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(self.clip_model)
        self.text_features = self.get_text_features(classnames)
        self.logit_scale = self.clip_model.logit_scale
        # self.v_linear_weight = self.clip_model.visual.attnpool.v_proj.weight
        # self.v_linear_bias = self.clip_model.visual.attnpool.v_proj.bias
        # self.c_linear_weight = self.clip_model.visual.attnpool.c_proj.weight
        # self.c_linear_bias = self.clip_model.visual.attnpool.c_proj.bias

    def forward(self, image, ):
        image_features, _ = self.clip_model.encode_image(image.type(self.dtype))      # [bs, 196, 512]

            # image_features = self.clip_model.encode_image(image.type(self.dtype))         # [bs, 512, 14, 14]
            # b, c, h, w = image_features.shape
            # image_features = image_features.reshape(b, c, h * w).permute(0, 2, 1)
            # image_features = F.linear(image_features, self.v_linear_weight, self.v_linear_bias)
            # image_features = F.linear(image_features, self.c_linear_weight, self.c_linear_bias)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)                     # [80, 512]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        coefficient = logit_scale * image_features @ text_features.t()                            # [bs, 196, 80]
        coefficient = F.softmax(coefficient, dim=1).permute(0, 2, 1)                 # [bs, 80, 196]
        sd_features = coefficient @ image_features                                   # [bs, 80, 512]

        logit_scale = self.logit_scale.exp()

        logits = sd_features * text_features
        logits = logits.sum(dim=-1)

        # logits = sd_features @ text_features.t()                                     # [bs, 80, 80]
        # logits = torch.diagonal(logits, dim1=-2, dim2=-1)                            # [bs, 80]                        
        
        return logits

    def get_text_features(self, classnames):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.clip_model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)
