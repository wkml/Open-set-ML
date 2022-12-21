import os
from pycocotools.coco import COCO
import json
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torchvision.transforms as transforms
import torch
from model.clip_base_sd import CLIP_SD
import argparse
import json
import cv2


test_data_dir='/data/public/coco2014/val2014'
test_list='/data/public/coco2014/annotations/instances_val2014.json'
with open('./data/coco/category.json', 'r') as load_category:
    category_map = json.load(load_category)
coco = COCO(test_list)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
scale_size = 512
crop_size = 448
test_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                          transforms.CenterCrop(crop_size),
                                          transforms.ToTensor(),
                                          ])

parser = argparse.ArgumentParser()
parser.add_argument('--backbone_name',default='RN101')
parser.add_argument('--crop_size',default=448)
args = parser.parse_args([])
graph_file='./data/coco/prob_train.npy'
word_file='./data/coco/vectors.npy'
with open('./data/coco/category_name.json', 'r') as load_category:
        category_map = json.load(load_category)

model = CLIP_SD(args=args,
                classnames=category_map,
                image_feature_dim=2048,
                num_classes=80,
                word_feature_dim=512,
)

checkpoint = torch.load('exp/checkpoint/SD-CLIP_BASE-COCO-exp3.1-lr3e-4-bs16/Checkpoint_Best.pth', map_location=torch.device('cuda'))
best_prec = checkpoint['best_mAP']
model.load_state_dict(checkpoint['state_dict'])

ids = list(coco.imgs.keys())
img_id = ids[0]
path = coco.loadImgs(img_id)[0]['file_name']
image = Image.open(os.path.join(test_data_dir, path)).convert('RGB')
input = test_data_transform(image).unsqueeze(0).cuda()

rgb_img = cv2.imread(os.path.join(test_data_dir, path), 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (448, 448))
rgb_img = np.float32(rgb_img) / 255

target_layers = [model.clip_model.visual.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
targets = [ClassifierOutputTarget(3)]
grayscale_cam = cam(input_tensor=input,targets=targets)
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(f'2_cam.jpg', cam_image)