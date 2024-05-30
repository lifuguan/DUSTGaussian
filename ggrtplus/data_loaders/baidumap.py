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

import enum
import os
import numpy as np
import imageio
import torch
import sys
import cv2
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import downsample_gaussian_blur

class BaiduMapDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.num_source_views = args.num_source_views
        self.folder_path = os.path.join('data/baidu_map')
        self.dataset_name = 'waymo'
        self.image_size = (336,512)
        self.ORIGINAL_SIZE = (389,573)

        self.is_train = True if mode == 'train' else False
        scenes = ['004']
        print("loading {} for {}".format(scenes, mode))
        print(f'[INFO] num scenes: {len(scenes)}')
        scene_path = os.path.join(self.folder_path, scenes[0])
        self.rgb_files = sorted(os.listdir(os.path.join(scene_path, "images")))
        self.rgb_files = [os.path.join(scene_path, "images", f) for f in self.rgb_files]
        self.extrinsics_files = [f.replace("images", "extrinsics").replace("jpg", "txt") for f in self.rgb_files]
        
        intrinsic = np.loadtxt(os.path.join(scene_path, "intrinsic.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        fx, fy = (
            fx * self.image_size[1] / self.ORIGINAL_SIZE[1],
            fy * self.image_size[0] / self.ORIGINAL_SIZE[0],
        )
        cx, cy = (
            cx * self.image_size[1] / self.ORIGINAL_SIZE[1],
            cy * self.image_size[0] / self.ORIGINAL_SIZE[0],
        )
        self.intrinsic = np.array([[fx, 0, cx,0], [0, fy, cy,0], [0, 0, 1, 0], [0, 0, 0, 1]])

        self.depth_range = torch.tensor([0.1, 100.0])

    def __len__(self):
        return len(self.rgb_files) - 6

    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized

    def __getitem__(self, idx):
        idx = idx + 1
        target_img_path = self.rgb_files[idx]
        rgb = imageio.imread(target_img_path).astype(np.float32) / 255.
        rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.image_size[1] / self.ORIGINAL_SIZE[1]), (self.image_size[1],self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        extrinsic_img_path = self.extrinsics_files[idx]
        extrinsic = torch.from_numpy(np.loadtxt(extrinsic_img_path)).unsqueeze(0)

        pil_path = [self.rgb_files[idx - 1], self.rgb_files[idx], self.rgb_files[idx+1]]
        if self.is_train is True:
            src_idxs = [idx-1, idx+1]
        else:
            src_idxs = [idx, idx+1]

        src_rgbs, src_extrinsics, src_intrinsics = [], [], []
        for src_idx in src_idxs:
            src_img_path = self.rgb_files[src_idx]
            src_extrinsic_img_path = self.extrinsics_files[src_idx]
            src_rgb = imageio.imread(src_img_path).astype(np.float32) / 255.
            src_rgb = cv2.resize(downsample_gaussian_blur(
                src_rgb, self.image_size[1] / self.ORIGINAL_SIZE[1]), (self.image_size[1],self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            src_extrinsic = np.loadtxt(src_extrinsic_img_path)

            src_rgbs.append(torch.from_numpy(src_rgb[..., :3]).permute(2, 0, 1).unsqueeze(0))
            src_extrinsics.append(torch.from_numpy(src_extrinsic).unsqueeze(0))
            src_intrinsics.append(torch.from_numpy(self.intrinsic).unsqueeze(0))   
        src_rgbs = torch.cat(src_rgbs, dim=0)
        src_extrinsics = torch.cat(src_extrinsics, dim=0)
        src_intrinsics = torch.cat(src_intrinsics, dim=0)

        src_intrinsics = self.normalize_intrinsics(src_intrinsics[:,:3,:3].float(), self.image_size)
        intrinsic = self.normalize_intrinsics(torch.from_numpy(self.intrinsic[:3,:3]).unsqueeze(0).float(), self.image_size)


        # Resize the world to make the baseline 1.
        if src_extrinsics.shape[0] == 2:
            a, b = src_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            if scale < 0.001:
                print(
                    f"Skipped because of insufficient baseline "
                    f"{scale:.6f}"
                )
            src_extrinsics[:, :3, 3] /= scale
            extrinsic[:, :3, 3] /= scale
        else:
            scale = 1
        return {
                'idx': idx,
                "dust_img": pil_path,
                "context": {
                        "extrinsics": src_extrinsics.float(),
                        "intrinsics": src_intrinsics,
                        "image": src_rgbs,
                        "near":  self.depth_range[0].repeat(self.num_source_views) / scale,
                        "far": self.depth_range[1].repeat(self.num_source_views) / scale,
                        "index": torch.tensor(src_idxs),
                },
                "target": {
                        "extrinsics": extrinsic.float(),
                        "intrinsics": intrinsic,
                        "image": torch.from_numpy(rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                        "near": self.depth_range[0].unsqueeze(0) / scale,
                        "far": self.depth_range[1].unsqueeze(0) / scale,
                        "index": torch.tensor(idx),
                }}