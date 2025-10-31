import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_file_path, os.pardir))
sys.path.append(root_path)

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import timm
import numpy as np
import pandas as pd

from PIL import Image
from models import ens_models, ens_path

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = ['resnet50', 'densenet121'] 
inv_model_paper = ['inception_v3', 'inception_resnet_v2'] 
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224']
ens_model_paper = ['IncV3Ens3', 'IncV3Ens4', 'IncResV2Ens'] 

cnn_model_paper_d = ['inception_v3', 'inception_v4','inception_resnet_v2','resnet101','vgg19','densenet121']
vit_model_paper_d = ['vit_base_patch16_224','levit_256', 'pit_b_224','cait_s24_224','convit_base', 'tnt_s_patch16_224', 'visformer_small']

cnn_model_paper_g = ['vgg16', 'resnet18', 'resnet50', 'densenet121', 'mobilenet_v2']
inv_model_paper_g = ['inception_resnet_v2', 'inception_v3', 'inception_v4'] 
vit_model_paper_g = ['vit_base_patch16_224', 'pit_b_224'] 
ens_model_paper_g = ['IncV3Ens3', 'IncV3Ens4', 'IncResV2Ens'] 


cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']

vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

def load_pretrained_model(cnn_model=[], inv_model=[], vit_model=[], ens_model=[]):
    # for model_name in cnn_model:
    #     if model_name in models.__dict__:
    #         yield model_name, models.__dict__[model_name](weights="DEFAULT")
    #     else: 
    #         yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")
    for model_name in inv_model:
        yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in vit_model:
        yield model_name, timm.create_model(model_name, pretrained=True)
    for model_name in ens_model:
        model = ens_models[model_name](weight_file=ens_path[model_name], aux_logits=False)
        yield model_name, model

def load_model(model_name):
    """
    The model Loading stage, which should be overridden when surrogate model is customized (e.g., DSM, SETR, etc.)
    Prioritize the model in torchvision.models, then timm.models

    Arguments:
        model_name (str/list): the name of surrogate model in model_list in utils.py

    Returns:
        model (torch.nn.Module): the surrogate model wrapped by wrap_model in utils.py
    """
    def load_single_model(model_name):
        if model_name in models.__dict__.keys():
            print('=> Loading model {} from torchvision.models'.format(model_name))
            model = models.__dict__[model_name](weights="DEFAULT")
        elif model_name in timm.list_models():
            print('=> Loading model {} from timm.models'.format(model_name))
            model = timm.create_model(model_name, pretrained=True)
        else:
            raise ValueError('Model {} not supported'.format(model_name))
        return wrap_model(model.eval().cuda())

    if isinstance(model_name, list):
        return EnsembleModel([load_single_model(name) for name in model_name])
    else:
        return load_single_model(model_name)

def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    if hasattr(model, 'default_cfg'):
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, targeted=False, eval = False):
        self.dir = output_dir if eval else input_dir
        if not eval:
            self.dir = os.path.join(self.dir, 'images')
        self.csv = pd.read_csv(os.path.join(input_dir, 'images.csv'))
        self.targeted = targeted

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        label = [Truelabel, TargetClass] if self.targeted else Truelabel

        # load img
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).resize((img_height, img_width)).convert('RGB')
        image = np.array(pil_img).astype(np.float32)/255
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, label, ImageID

    def __len__(self):
        return len(self.csv)