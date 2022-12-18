import torch.nn as nn
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from .element_wise_layer import Element_Wise_Layer
from .prompt_learner import PromptLearner, TextEncoder

class COOP(nn.Module):
    def __init__(self, args, classnames):
        super(COOP, self).__init__()
        self.clip_model = load_clip_to_cpu(args).float()
        self.prompt_learner = PromptLearner(args, classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

    def forward(self, image):
        _, image_features = self.image_encoder(image.type(self.dtype))       #[bs, 2048, H, W]
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # SD

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
