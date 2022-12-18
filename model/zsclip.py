import torch
import torch.nn as nn
from utils.checkpoint import load_clip_to_cpu
from clip import clip

class ZSCLIP(nn.Module):
    def __init__(self, args, classnames):
        super(ZSCLIP, self).__init__()
        self.clip_model = load_clip_to_cpu(args).float()
        self.image_encoder = self.clip_model.visual
        self.text_features = self.get_text_features(classnames)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

    def forward(self, image):
        _, image_features = self.image_encoder(image.type(self.dtype))       #[bs, 512, H, W]
        text_features = self.text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_text_features(self, classnames):
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        text_features = self.clip_model.encode_text(prompts)
        return nn.Parameter(text_features, requires_grad=False)
    