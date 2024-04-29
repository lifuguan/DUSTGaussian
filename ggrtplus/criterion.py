# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
from utils import img2mse

from ggrtplus.geometry.camera import Camera
from ggrtplus.geometry.depth import calc_smoothness, inv2depth
from ggrtplus.pose_util import Pose


class MaskedL2ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        if 'mask' in outputs:
            pred_mask = outputs['mask'].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss


def pseudo_huber_loss(residual, scale=10):
    trunc_residual = residual / scale
    return torch.sqrt(trunc_residual * trunc_residual + 1) - 1


class FeatureMetricLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, target_rgb_feat, nearby_view_rgb_feat, mask=None):
        '''
        Args:
            target_rgb_feat: [n_rays, n_samples=1, n_views+1, d+3]
            nearby_view_rgb_feat: [n_rays, n_samples=1, n_views+1, d+3]
        '''
        if mask is None:
            l1_loss = nn.L1Loss(reduction='mean')
            # mse_loss = nn.MSELoss(reduction='mean')
            # loss = mse_loss(nearby_view_rgb_feat, target_rgb_feat)

            loss = l1_loss(nearby_view_rgb_feat, target_rgb_feat)
        
        else:
            feat_diff = target_rgb_feat - nearby_view_rgb_feat
            feat_diff_square = (feat_diff * feat_diff).squeeze(1)
            mask = mask.repeat(1, 1, 1).permute(2, 0, 1)
            n_views, n_dims = target_rgb_feat.shape[-2], target_rgb_feat.shape[-1]
            loss = torch.sum(feat_diff_square * mask) / (torch.sum(mask.squeeze(-1)) * n_views * n_dims + 1e-6)

            # feat_diff_huber = pseudo_huber_loss(feat_diff, scale=0.8).squeeze(1)
            # mask = mask.repeat(1, 1, 1).permute(2, 0, 1)
            # n_views, n_dims = target_rgb_feat.shape[-2], target_rgb_feat.shape[-1]
            # loss = torch.sum(feat_diff_huber * mask) / (torch.sum(mask.squeeze(-1)) * n_views * n_dims + 1e-6)
        
        return loss


def self_sup_depth_loss(inv_depth_prior, rendered_depth, min_depth, max_depth):
    min_disparity = 1.0 / max_depth
    max_disparity = 1.0 / min_depth
    valid = ((inv_depth_prior > min_disparity) & (inv_depth_prior < max_disparity)).detach()

    inv_rendered_depth = depth2inv(rendered_depth)

    loss_depth = torch.mean(valid * torch.abs(inv_depth_prior - inv_rendered_depth))

    return loss_depth


def sup_depth_loss(ego_motion_inv_depths, gt_depth, min_depth, max_depth):
    num_iters = len(ego_motion_inv_depths)
    total_loss = 0
    total_w = 0
    gamma = 0.85
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth

    gt_inv_depth = depth2inv(gt_depth)

    valid = ((gt_inv_depth > min_disp) & (gt_inv_depth < max_disp)).detach()

    for i, inv_depth in enumerate(ego_motion_inv_depths):
        w = gamma ** (num_iters - i - 1)
        total_w += w

        loss_depth = torch.mean(valid * torch.abs(gt_inv_depth - inv_depth.squeeze(0)))
        loss_i = loss_depth
        total_loss += w * loss_i
    loss = total_loss / total_w
    return loss



def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def shuffle(input):
    # shuffle dim=1
    idx = torch.randperm(input[0].shape[1])
    for i in range(input.shape[0]):
        input[i] = input[i][:, idx].view(input[i].shape)

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def loss_depth_grad(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = img_grad_x / (torch.abs(img_grad_x) + 1e-6)
    weight_y = img_grad_y / (torch.abs(img_grad_y) + 1e-6)

    depth_grad_x = depth[:, :, :, :-1] - depth[:, :, :, 1:]
    depth_grad_y = depth[:, :, :-1, :] - depth[:, :, 1:, :]
    grad_x = depth_grad_x / (torch.abs(depth_grad_x) + 1e-6)
    grad_y = depth_grad_y / (torch.abs(depth_grad_y) + 1e-6)

    loss = l1_loss(grad_x, weight_x) + l1_loss(grad_y, weight_y)
    return loss


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def margin_l1_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask].abs()).mean()
    else:
        return ((network_output - gt)[mask].abs()).mean(), mask
    

def kl_loss(input, target):
    input = F.log_softmax(input, dim=-1)
    target = F.softmax(target, dim=-1)
    return F.kl_div(input, target, reduction="batchmean")

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_l1_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_l1_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def margin_ssim(img1, img2, window_size=11, size_average=True):
    result = ssim(img1, img2, window_size, False)
    print(result.shape)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




########################################################################################################################

def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same

    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape

    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True


def interpolate_image(image, shape, mode='bilinear', align_corners=True):
    """
    Interpolate an image to a different resolution

    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Image to be interpolated
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    image : torch.Tensor [B,?,H,W]
        Interpolated image
    """
    # Take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:]
    # If the shapes are the same, do nothing
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        # Interpolate image to match the shape
        return torch.nn.functional.interpolate(image, size=shape, mode=mode,
                                 align_corners=align_corners)

def match_scales(image, targets, num_scales,
                 mode='bilinear', align_corners=True):
    """
    Interpolate one image to produce a list of images with the same shape as targets

    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Input image
    targets : list of torch.Tensor [B,?,?,?]
        Tensors with the target resolutions
    num_scales : int
        Number of considered scales
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation

    Returns
    -------
    images : list of torch.Tensor [B,?,?,?]
        List of images with the same resolutions as targets
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(interpolate_image(
                image, target_shape, mode=mode, align_corners=align_corners))
    # Return scaled images
    return images

########################################################################################################################


def view_synthesis(ref_image, depth, ref_cam, cam, mode='bilinear', padding_mode='zeros'):
    """
    Synthesize an image from another plus a depth map.

    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : Camera
        Camera class for the reference image
    cam : Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation

    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = cam.reconstruct(depth, frame='w')
    # Project world points onto reference camera
    ref_coords = ref_cam.project(world_points, frame='w')
    
    # View-synthesis given the projected reference points
    return torch.nn.functional.grid_sample(ref_image, ref_coords, mode=mode,
                             padding_mode=padding_mode, align_corners=True)


########################################################################################################################

def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

########################################################################################################################

class LossBase(nn.Module):
    """Base class for losses."""
    def __init__(self):
        """Initializes logs and metrics dictionaries"""
        super().__init__()
        self._logs = {}
        self._metrics = {}

########################################################################################################################

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def metrics(self):
        """Return metrics."""
        return self._metrics

    def add_metric(self, key, val):
        """Add a new metric to the dictionary and detach it."""
        self._metrics[key] = val.detach()

########################################################################################################################


class MultiViewPhotometricDecayLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them

    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scales to consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, ssim_loss_weight=0.85, smooth_loss_weight=0.01,
                 C1=1e-4, C2=9e-4, photometric_reduce_op='min', disp_norm=True, clip_loss=0.5,
                 padding_mode='zeros', automask_loss=True):
        super().__init__()
        
        self.n = 1
        self.ssim_loss_weight = ssim_loss_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

########################################################################################################################

    def warp_ref_image(self, inv_depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.

        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,H,W]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation

        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()
        
        # Generate cameras for all scales
        _, _, DH, DW = inv_depths.shape
        scale_factor = DW / float(W)
        cams = Camera(K=K.float()).scaled(scale_factor).to(device)
        ref_cams = Camera(K=ref_K.float(), Twc=pose).scaled(scale_factor).to(device)
        
        # View synthesis
        depths = inv2depth(inv_depths)
        # depths = inv_depths
        ref_warped = view_synthesis(ref_image, depths, ref_cams, cams, padding_mode=self.padding_mode)
        
        # Return warped reference image
        return ref_warped

########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss

        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter

        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_photometric_loss(self, photometric_losses):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                return torch.cat(losses, 1).min(1, True)[0].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))
        # Reduce photometric loss
        
        # photometric_loss = sum([reduce_function(photometric_losses[i])
                                # for i in range(self.n)]) / self.n
        
        gamma = 0.85
        photometric_loss = [reduce_function(photometric_losses[i]) for i in range(self.n)]
        photometric_loss_total = 0.0
        for i in range(self.n):
            i_weight = gamma**(self.n - i - 1)
            photometric_loss_total += i_weight * photometric_loss[i]
        photometric_loss = photometric_loss_total
        
        # Store and return reduced photometric loss
        self.add_metric('photometric_loss', photometric_loss)
        return photometric_loss

########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

########################################################################################################################

    def warp_image(self, image, ref_imgs, inv_depths, K, ref_Ks, poses):
        
        # Loop over all reference images
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)
        num_views = ref_imgs.shape[0]

        ref_warpeds = []
        for j in range(num_views):
            ref_image, pose = ref_imgs[j].unsqueeze(0), poses[j]
            # print(f'ref_image shape: {ref_image.shape}')
            # Calculate warped images
            ref_warpeds.append(self.warp_ref_image(inv_depths[j], ref_image, K, ref_Ks[j], pose.unsqueeze(0)))
        warped_image = torch.cat(ref_warpeds)
        return warped_image

########################################################################################################################

    def forward(self, image, ref_imgs, inv_depths, K, ref_Ks, poses):
        """
        Calculates training photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_imgs : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor n_iters*[B,1,H,W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        poses : torch Tensor [n_views, n_iters, 6]
            Camera transformation between original and ref_imgs
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = len(inv_depths)
        
        # Loop over all reference images
        photometric_losses = [[] for _ in range(self.n)]
        images = match_scales(image, inv_depths, self.n)
        num_views = ref_imgs.shape[0]

        for j in range(num_views):
            ref_image, pose = ref_imgs[j].unsqueeze(0), Pose.from_vec(poses[j])
            # print(f'ref_image shape: {ref_image.shape}')
            # Calculate warped images
            ref_warped = self.warp_ref_image(inv_depths, ref_image, K, ref_Ks[j], pose)
            
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])
            
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(ref_image, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])
        
        # Calculate reduced photometric loss
        loss = self.reduce_photometric_loss(photometric_losses)
        
        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            loss += self.calc_smoothness_loss(inv_depths, images)
        
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################