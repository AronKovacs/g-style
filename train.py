#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import open_clip

import numpy as np

from pathlib import Path

import nnfm_utils

def image_loader(path):
    image=Image.open(path)
    loader=transforms.Compose([transforms.ToTensor()])

    image=loader(image).unsqueeze(0)
    return image.to('cuda', torch.float)

def bcwh_to_bwhc(data):
    return torch.permute(data, (0, 2, 3, 1))

def bwhc_to_bcwh(data):
    return torch.permute(data, (0, 3, 1, 2))

class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
    
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to('cuda')
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to('cuda')

        self.mean = torch.tensor(cnn_normalization_mean).view(-1, 1, 1)
        self.std = torch.tensor(cnn_normalization_std).view(-1, 1, 1)
        
        self.req_features= ['0','5','10','19','28']
        self.model=models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if str(layer_num) in self.req_features:
                features.append(x)
                
        return features

def calc_content_loss(gen_feat,orig_feat):
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l

def calculate_content_loss(generated_features, content_features):
    content_loss = 0.0
    for g, c in zip(generated_features, content_features):
        content_loss += torch.nn.functional.mse_loss(g, c)
    return content_loss

def calculate_total_variation_loss(image):
    width_variance = torch.sum(torch.pow(image[:,:,:,:-1] - image[:,:,:,1:], 2))
    height_variance = torch.sum(torch.pow(image[:,:,:-1,:] - image[:,:,1:,:], 2))
    loss = width_variance + height_variance
    return loss

def get_feats(x, vgg16, vgg16_normalize, layers=[]):
	x = vgg16_normalize(x)
	final_ix = max(layers)
	outputs = []

	for ix, layer in enumerate(vgg16.features):
		x = layer(x)
		if ix in layers:
			outputs.append(x)

		if ix == final_ix:
			break
	return outputs

class CLIPLoss(torch.nn.Module):
  def __init__(self, text_prompts=[], image_prompts=[], n_cuts=16):
    super(CLIPLoss, self).__init__()
    
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
    self.clip_model = clip_model
    self.clip_model_input_size = 224
    self.preprocess = transforms.Compose([
        transforms.Resize(size=self.clip_model_input_size, max_size=None, antialias=None),
        transforms.CenterCrop(size=(self.clip_model_input_size, self.clip_model_input_size)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    self.clip_model.to('cuda')
    self.clip_model.eval()
    
    self.target_embeds = []
    with torch.no_grad():
      for text_prompt in text_prompts:
        tokenized_text = open_clip.tokenize([text_prompt]).to('cuda')
        self.target_embeds.append(clip_model.encode_text(tokenized_text))
      for image_prompt in image_prompts:
        image_embed = clip_model.encode_image(self.preprocess(image_prompt))
        self.target_embeds.append(image_embed)

    self.target_embeds = torch.cat(self.target_embeds)

    self.n_cuts = n_cuts

  def forward(self, input):
    if self.n_cuts > 1:
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.clip_model_input_size)
        cutouts = []
        for _ in range(self.n_cuts):
            size = int(torch.rand([]) * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.clip_model_input_size))
        input = torch.cat(cutouts)

    input_embed = self.clip_model.encode_image(self.preprocess(input))  
    input_normed = torch.nn.functional.normalize(input_embed.unsqueeze(1), dim=-1)
    embed_normed = torch.nn.functional.normalize(self.target_embeds.unsqueeze(0), dim=-1)
    dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)

    return dists.mean()

vgg16 = models.vgg16(pretrained=True).eval().to('cuda')
vgg16_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def projected_area(scaling):
    s0 = scaling[0]
    s1 = scaling[1]
    s2 = scaling[2]
    
    a = 0
    b = 0
    if s0 < s1 and s0 < s2:
        a = s1
        b = s2
    elif s1 < s0 and s1 < s2:
        a = s0
        b = s2
    else:
        a = s0
        b = s1
    return a * b

def shorten_scaling(scaling, threshold):
    s0 = scaling[0]
    s1 = scaling[1]
    s2 = scaling[2]
    
    a = 0
    b = 0
    if s0 < s1 and s0 < s2:
        a = s1
        b = s2
    elif s1 < s0 and s1 < s2:
        a = s0
        b = s2
    else:
        a = s0
        b = s1
    
    a, b = max(a, b), min(a, b)
    ratio = a / b
    if ratio > threshold:
        new_a = b + (a - b) * 0.5

        if scaling[0] == a:
            scaling[0] = new_a
        elif scaling[1] == a:
            scaling[1] = new_a
        else:
            scaling[2] = new_a


def training(dataset, opt, pipe, checkpoint):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, int(dataset.starting_iter))
    # copy everything, but we mostly care about the shape
    gaussians.copy_features_primary_to_secondary()
    with torch.no_grad():
        gaussians._features_rest[:] = 0.0

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    style_name = Path(dataset.path_style).stem
    style_image = image_loader(dataset.path_style)[:, :3]

    clip_loss_fn = CLIPLoss(
        image_prompts=[style_image],
    )
    vgg16_style_feats = [x.detach() for x in get_feats(style_image, vgg16, vgg16_normalize, nnfm_utils.nnfm_all_layers())]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1

    scaling = gaussians._scaling.cpu().detach().numpy()
    print(scaling.shape)
    
    n_views = len(scene.getTrainCameras())
    pretrain_until_at_least = None
    dont_split_when_above = 3250000
    
    if dataset.forward_facing:
        print('Setting params for a forward facing scene')
        clip_weight = 10
        nnfm_weight = 100
        content_weight = 0.05
        tv_weight = 0.0001

        n_retraining = 5
        
        split_std_threshold = 1.0
        split_std_multiplier = 1.0
        split_max_n = 0.10
        split_by = 8
        split_scaling_factor = 0.8
        retrain_iterations = max(n_views * 5, 2000)
        shorten_scaling_iter = 250
        shorten_scaling_max_iter = 1000
        shorten_scaling_threshold = 1.5
        stylization_size = 1024
        stylization_iterations = max(n_views * 15, 450)
        stylization_gradient_accum_start = n_views * 2
        stylization_split_iterations = [n_views * 3]
        stylization_densify_by = 4
        stylization_densify_scaling_factor = 4
        stylization_densify_n = 0.05
        stylization_retrain_iterations = retrain_iterations
        color_matching_iterations = max(n_views * 4, 1000)
        
        stylization_lr_start = 0.1
        stylization_lr_end = 0.01
        stylization_decay = -np.log(stylization_lr_end/stylization_lr_start)/stylization_iterations
        color_matching_start_iteration = int(1 * stylization_iterations)
    else:
        print('Setting params for a 360 scene')
        
        clip_weight = 10
        nnfm_weight = 10
        content_weight = 0.05
        tv_weight = 0.0001

        n_retraining = 10
        
        split_std_threshold = 1.1
        split_std_multiplier = 1.125
        split_max_n = 0.05
        split_by = 4
        split_scaling_factor = 2
        retrain_iterations = max(n_views * 5, 2000)
        shorten_scaling_iter = 250
        shorten_scaling_max_iter = 1000
        shorten_scaling_threshold = 1.5
        stylization_size = 1024
        stylization_iterations = max(n_views * 15, 300)
        stylization_gradient_accum_start = n_views * 2
        stylization_split_iterations = [n_views * 3, n_views, n_views * 7]
        stylization_densify_by = 4
        stylization_densify_scaling_factor = 2
        stylization_densify_n = 0.02
        stylization_retrain_iterations = retrain_iterations
        color_matching_iterations = max(n_views * 4, 1000)
        
        stylization_lr_start = 0.01
        stylization_lr_end = 0.005
        stylization_decay = -np.log(stylization_lr_end/stylization_lr_start)/stylization_iterations
        color_matching_start_iteration = int(1 * stylization_iterations)
        
    starting_iter = 100000
    preprocess_color_matching_iter = 100001
    if int(dataset.starting_iter) == starting_iter:
        n_retraining = 0
        print('Loaded a retrained dataset')
    else:
        print('Retraining a dataset')

    gaussians.active_sh_degree = 0

    print('Number of Gaussians:', gaussians._features_dc.shape[0])

    if pretrain_until_at_least is not None:
        n_retraining = 10000000 # a large number
    for retraining in range(n_retraining):
        if gaussians._features_dc.shape[0] > dont_split_when_above:
            break
        if pretrain_until_at_least is not None and gaussians._features_dc.shape[0] > pretrain_until_at_least:
            break
    
        progress_bar = tqdm(range(0, retrain_iterations), desc=f"Retraining progress {retraining + 1}")

        with torch.no_grad():
            if split_max_n > 0.0:
                scaling = gaussians.get_scaling.cpu().detach().numpy()
                n_gaussians = gaussians._features_dc.shape[0]
            
                projected_areas = np.zeros(scaling.shape[0], np.float32)
                for i in range(scaling.shape[0]):
                    projected_areas[i] = projected_area(scaling[i])
                
                areas_mean = np.mean(projected_areas)
                areas_std = np.std(projected_areas)
                
                std_split_threshold = areas_mean + areas_std * split_std_threshold * (split_std_multiplier ** retraining)
                split_counter = np.count_nonzero(projected_areas > std_split_threshold)
                print('split_counter', split_counter)
                if split_counter > split_max_n * n_gaussians:
                    sorted_projected_areas = np.sort(projected_areas)
                    percentage_split_threshold = sorted_projected_areas[n_gaussians - int(split_max_n * n_gaussians)]
                    split_threshold = percentage_split_threshold
                else:
                    split_threshold = std_split_threshold
                
                gaussians_to_split = torch.empty(n_gaussians, dtype=torch.bool, device='cpu')
                for i in range(n_gaussians):
                    gaussians_to_split[i] = bool(projected_areas[i] > split_threshold)
                gaussians_to_split = gaussians_to_split.cuda()
                gaussians.densify_and_split_with_mask(gaussians_to_split, N=split_by, scaling_factor=split_scaling_factor)

        for iteration in range(1, retrain_iterations + 1):
            iter_start.record()

            if iteration % shorten_scaling_iter == 0 and iteration <= shorten_scaling_max_iter:
                with torch.no_grad():
                    scaling = gaussians.get_scaling.cpu().detach().numpy()
                    for i in range(scaling.shape[0]):
                        shorten_scaling(scaling[i], shorten_scaling_threshold)
                    gaussians._scaling[:, :] = gaussians.scaling_inverse_activation(torch.from_numpy(scaling))

            gaussians.optimizer.zero_grad()
            gaussians.update_learning_rate(30000 - retrain_iterations + iteration)

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, primary_features=True)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"].unsqueeze(0), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda().unsqueeze(0)
            Ll1 = l1_loss(image, gt_image)
            content_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            loss = content_loss

            loss.backward()
            gaussians.optimizer.step()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == retrain_iterations:
                    progress_bar.close()
                    
    print('Number of Gaussians after pretraining:', gaussians._features_dc.shape[0])

    if n_retraining > 0:
        print('Saving retrained Gaussians')
        scene.save(100000, primary_features=True)
        
    gaussians.copy_features_primary_to_secondary()
        
    if int(dataset.starting_iter) >= preprocess_color_matching_iter:
        color_matching_iterations = 0
    else:
        gaussians._features_secondary_dc.data[:, :, 0] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
        gaussians._features_secondary_dc.data[:, :, 1] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
        gaussians._features_secondary_dc.data[:, :, 2] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
    
        progress_bar = tqdm(range(0, color_matching_iterations), desc=f"Color matching progress")        
        color_matching_optimizer = torch.optim.Adam([gaussians._features_secondary_dc], lr=0.01)
        
    for iteration in range(1, color_matching_iterations + 1):
        iter_start.record()

        color_matching_optimizer.zero_grad()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, primary_features=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"].unsqueeze(0), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda().unsqueeze(0)
        gt_image = bwhc_to_bcwh(nnfm_utils.match_colors_for_image_set(bcwh_to_bwhc(gt_image), bcwh_to_bwhc(style_image)[0])[0])
        
        Ll1 = l1_loss(image, gt_image)
        content_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = content_loss

        loss.backward()
        color_matching_optimizer.step()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == color_matching_iterations:
                progress_bar.close()
                
    if color_matching_iterations > 0:
        print('Saving recolored Gaussians')
        scene.save(100001, primary_features=False)
    
    progress_bar = tqdm(range(0, stylization_iterations), desc=f"Stylization progress")

    gaussian_optimizer = torch.optim.Adam([gaussians._features_secondary_dc], lr=stylization_lr_start)
    gaussians_grad = torch.zeros((gaussians._features_secondary_dc.shape[0]), device='cuda')
    gaussians_denom = torch.zeros((gaussians._features_secondary_dc.shape[0]), device='cuda')

    for iteration in range(1, stylization_iterations + 1):
        iter_start.record()
        
        gaussian_optimizer.param_groups[0]['lr'] = stylization_lr_start * np.exp(-stylization_decay * iteration)

        gaussian_optimizer.zero_grad()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda().unsqueeze(0)
        gt_image = bwhc_to_bcwh(nnfm_utils.match_colors_for_image_set(bcwh_to_bwhc(gt_image), bcwh_to_bwhc(style_image)[0])[0])
        
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, primary_features=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"].unsqueeze(0), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if stylization_size is not None:
            with torch.no_grad():
                old_size = (image.shape[2], image.shape[3])
                size_ratio = stylization_size / old_size[1]
                new_size = (int(old_size[0] * size_ratio), stylization_size)
        else:
            new_size = (image.shape[2], image.shape[3])
        
        with torch.no_grad():
            downsampled_gt_image = torch.nn.functional.interpolate(gt_image, size=new_size, mode='bilinear', antialias=True)
            downsampled_gt_image_feats = [x.detach() for x in get_feats(downsampled_gt_image, vgg16, vgg16_normalize, nnfm_utils.nnfm_all_layers())]
        downsampled_image = torch.nn.functional.interpolate(image, size=new_size, mode='bilinear', antialias=True)
        downsampled_image_feats = get_feats(downsampled_image, vgg16, vgg16_normalize, nnfm_utils.nnfm_all_layers())
        
        clip_loss = clip_loss_fn(downsampled_image) * clip_weight if clip_weight != 0 else 0
        nnfm_loss = nnfm_utils.calculate_nnfm_loss(downsampled_image_feats, vgg16_style_feats) * nnfm_weight if nnfm_weight != 0 else 0
        content_loss = calculate_content_loss(downsampled_image_feats, downsampled_gt_image_feats) * content_weight if content_weight != 0 else 0
        total_variation_loss = calculate_total_variation_loss(image) * tv_weight if tv_weight != 0 else 0
        
        loss = clip_loss + nnfm_loss + content_loss + total_variation_loss
        
        loss.backward()
        gaussian_optimizer.step()

        with torch.no_grad():
            if iteration % 5000 == 0 and iteration != stylization_iterations:
                print('Saving scene', 100000 + iteration)
                scene.save(100000 + iteration, primary_features=False)

        with torch.no_grad():
            if iteration >= stylization_gradient_accum_start:
                visible_gaussians = torch.count_nonzero(visibility_filter)
                gaussians_grad[visibility_filter] += torch.norm(torch.reshape(gaussians._features_secondary_dc.grad[visibility_filter], (visible_gaussians, 3)), dim=-1)
                gaussians_denom[visibility_filter] += 1

            if iteration in stylization_split_iterations and gaussians._features_dc.shape[0] <= dont_split_when_above:
                gaussians_grad /= gaussians_denom + 1

                gaussians_grad_threshold = np.sort(gaussians_grad.cpu().detach().numpy())[int((1.0 - stylization_densify_n) * gaussians_grad.shape[0])]
                split_mask = gaussians_grad > gaussians_grad_threshold
                print('splitting from', gaussians._features_secondary_dc.shape[0])
                gaussians.densify_and_split_with_mask(split_mask, N=stylization_densify_by, scaling_factor=stylization_densify_scaling_factor)
                print('splitting to', gaussians._features_secondary_dc.shape[0])

                gaussian_optimizer = torch.optim.Adam([gaussians._features_secondary_dc], lr=gaussian_optimizer.param_groups[0]['lr'])
                gaussians_grad = torch.zeros((gaussians._features_secondary_dc.shape[0]), device='cuda')
                gaussians_denom = torch.zeros((gaussians._features_secondary_dc.shape[0]), device='cuda')

        if iteration in stylization_split_iterations and gaussians._features_dc.shape[0] <= dont_split_when_above:
            for retrain_iteration in range(1, stylization_retrain_iterations + 1):
                if retrain_iteration % shorten_scaling_iter == 0 and retrain_iteration <= shorten_scaling_max_iter:
                    with torch.no_grad():
                        scaling = gaussians.get_scaling.cpu().detach().numpy()
                        for i in range(scaling.shape[0]):
                            shorten_scaling(scaling[i], shorten_scaling_threshold)
                        gaussians._scaling[:, :] = gaussians.scaling_inverse_activation(torch.from_numpy(scaling))

                gaussians.enable_geometry_learning(30000 - stylization_retrain_iterations + retrain_iteration)
                gaussians.optimizer.zero_grad()
                gaussians.update_learning_rate(30000 - stylization_retrain_iterations + retrain_iteration)

                # Pick a random Camera
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, primary_features=True)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"].unsqueeze(0), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                gt_image = viewpoint_cam.original_image.cuda().unsqueeze(0)
                Ll1 = l1_loss(image, gt_image)
                content_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                loss = content_loss

                loss.backward()
                gaussians.optimizer.step()

        if iteration == color_matching_start_iteration:
            viewpoint_stack_for_recoloring = scene.getTrainCameras().copy()
            recolored_viewpoint_renderings = []
            
            with torch.no_grad():
                pass
                for i in range(len(viewpoint_stack_for_recoloring)):
                    viewpoint_cam = viewpoint_stack_for_recoloring[i]        
                    bg = torch.rand((3), device="cuda") if opt.random_background else background

                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, primary_features=False)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"].unsqueeze(0), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    
                    recolored_image = bwhc_to_bcwh(nnfm_utils.match_colors_for_image_set(bcwh_to_bwhc(image), bcwh_to_bwhc(style_image)[0])[0])
                    recolored_viewpoint_renderings.append(recolored_image.cpu().detach())
                
                gaussians._features_secondary_dc.data[:, :, 0] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
                gaussians._features_secondary_dc.data[:, :, 1] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
                gaussians._features_secondary_dc.data[:, :, 2] = ((gaussians._features_secondary_dc.data[:, :, 0] + gaussians._features_secondary_dc.data[:, :, 1] + gaussians._features_secondary_dc.data[:, :, 2]) / 3.0)
                    
            color_matching_optimizer = torch.optim.Adam([gaussians._features_secondary_dc], lr=0.01)
            
            viewpoint_stack_for_recoloring = []
            for recoloring_iteration in range(1, color_matching_iterations + 1):
                iter_start.record()

                color_matching_optimizer.zero_grad()

                # Pick a random Camera
                if not viewpoint_stack_for_recoloring:
                    viewpoint_stack_for_recoloring = scene.getTrainCameras().copy()
                    recolored_renderings = recolored_viewpoint_renderings.copy()
                random_cam_id = randint(0, len(viewpoint_stack_for_recoloring)-1)
                viewpoint_cam = viewpoint_stack_for_recoloring.pop(random_cam_id)
                gt_image = recolored_renderings.pop(random_cam_id).cuda()

                bg = torch.rand((3), device="cuda") if opt.random_background else background

                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, primary_features=False)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"].unsqueeze(0), render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                
                Ll1 = l1_loss(image, gt_image)
                content_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                loss = content_loss

                loss.backward()
                color_matching_optimizer.step()

                iter_end.record()
                        
            #print("\n[ITER {}] Saving Gaussians".format(iteration))
            print('Saving recolored stylized Gaussians')
            scene.save(199999, primary_features=False)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == stylization_iterations:
                progress_bar.close()

    print("\n[ITER {}] Saving stylized Gaussians".format(iteration))
    scene.save(200000, primary_features=False)
    scene.save(200000, primary_features=False, iteration_prefix=style_name)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6009)
parser.add_argument('--debug_from', type=int, default=-1)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
parser.add_argument("--start_checkpoint", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)

print("Optimizing " + args.model_path)

# Initialize system state (RNG)
safe_state(args.quiet)

# Start GUI server, configure and run trainings
network_gui.init(args.ip, args.port)
torch.autograd.set_detect_anomaly(args.detect_anomaly)
training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)

# All done
print("\nTraining complete.")
