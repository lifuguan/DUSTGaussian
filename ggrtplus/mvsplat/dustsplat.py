from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn, optim

from ..global_cfg import get_cfg
from .types import Gaussians
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
import numpy as np
from .wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from .interpolatation import interpolate_extrinsics,interpolate_intrinsics


class dustSplat(nn.Module):
    encoder: nn.Module
    decoder: Decoder

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        encoder_visualizer: Optional[EncoderVisualizer],
    ) -> None:
        super().__init__()
         # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_visualizer = encoder_visualizer
        
        self.data_shim = get_data_shim(self.encoder)
        self.last_ref_gaussians = {}
            
        self.test_iteration = 0

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.gaussian_model.parameters(), lr=self.config.optimizer.lr)
        warm_up_steps = self.config.optimizer.warm_up_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)

    def forward(self, batch, features,cnns,depths,densities,global_step):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians= self.encoder(
            batch["context"], features, cnns ,depths, densities,global_step, 
            False, scene_names=batch["scene"],visualization_dump = None)
            
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth',
        )
        ret = {'rgb': output.color, 'depth': output.depth}
        if 'depth' in batch["target"].keys():
            target_gt = {'rgb': batch["target"]["image"], 'depth': batch["target"]["depth"]}
        else:
            target_gt = {'rgb': batch["target"]["image"]}
            
        if get_cfg().use_aux_loss is True:
            output_ref = self.decoder.forward(
                gaussians,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                depth_mode='depth',
            )
            ret_ref = {'rgb': output_ref.color, 'depth': output_ref.depth}
            if 'depth' in batch["context"].keys():
                target_gt_ref = {'rgb': batch["context"]["image"], 'depth': batch["context"]["depth"]}    
            else:
                target_gt_ref = {'rgb': batch["context"]["image"]}    
            return ret, target_gt, ret_ref, target_gt_ref
        else:
            return ret, target_gt, _, _




        # ret = {'rgb': output.color, 'depth': output.depth, "ex": extrinsics_pred}
        # target_gt = {'rgb': batch["target"]["image"], 'depth': batch["target"]["depth"],'ex': batch["context"]["extrinsics"]}
        # if get_cfg().use_aux_loss is True:
        #     output_ref = self.decoder.forward(
        #         gaussians,
        #         batch["context"]["extrinsics"],
        #         batch["context"]["intrinsics"],
        #         batch["context"]["near"],
        #         batch["context"]["far"],
        #         (h, w),
        #         depth_mode='depth',
        #     )
        #     ret_ref = {'rgb': output_ref.color, 'depth': output_ref.depth}
        #     target_gt_ref = {'rgb': batch["context"]["image"], 'depth': batch["context"]["depth"]}    
        #     return ret, target_gt, ret_ref, target_gt_ref
        # else:
        #     return ret, target_gt, _, _
        
    
    def inference(self, batch,  features,depths,densities,global_step):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        visualization_dump = {}

        # Run the model.
        gaussians,extrinsics_pred,rel_pose_iter,depths_iter = self.encoder(
            batch["context"],batch, features,depths,densities, global_step, 
            deterministic=True, 
            scene_names=batch["scene"],
            visualization_dump=visualization_dump,
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth',
        )
            
        ret = {'rgb': output.color, 'depth': output.depth,'ex':extrinsics_pred}
        target_gt = {'rgb': batch["target"]["image"], 'depth': batch["target"]["depth"],'ex': batch["context"]["extrinsics"]}
        return ret, target_gt, visualization_dump, gaussians