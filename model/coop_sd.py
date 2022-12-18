import torch.nn as nn
from utils.checkpoint import load_clip_to_cpu
from .semantic import semantic
from .element_wise_layer import Element_Wise_Layer
from .prompt_learner import PromptLearner, TextEncoder

class COOP_SD(nn.Module):
    def __init__(self, args, classnames,
                image_feature_dim=2048, num_classes=80, 
                word_feature_dim=512):
        super(COOP_SD, self).__init__()
        self.clip_model = load_clip_to_cpu(args).float()
        self.prompt_learner = PromptLearner(args, classnames, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.num_classes = num_classes
        self.word_feature_dim = word_feature_dim
        self.image_feature_dim = image_feature_dim
        self.dtype = self.clip_model.dtype
        self.word_semantic = semantic(num_classes= self.num_classes,
                                      image_feature_dim = self.image_feature_dim,
                                      word_feature_dim=self.word_feature_dim)

        self.classifiers = Element_Wise_Layer(self.num_classes, self.image_feature_dim)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))       #[bs, 2048, H, W]
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # SD
        sd_features = self.word_semantic(image_features, text_features)    # [bs, 80, 512]

        output = self.classifiers(sd_features)
        
        return output
