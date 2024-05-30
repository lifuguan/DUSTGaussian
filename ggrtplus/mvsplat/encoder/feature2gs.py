from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict
from .costvolume.ldm_unet.unet import UNetModel
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone.unimatch.backbone import ResidualBlock
from .costvolume.ldm_unet.unet import Upsample
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .common.plucker import plucker_embedding
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg
from ...global_cfg import get_cfg
# from .epipolar.conversions import relative_disparity_to_depth

def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    return 1 / ((1 - relative_disparity) * (disp_near - disp_far) + disp_far + eps)


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class EncoderDust2GS(Encoder[EncoderCostVolumeCfg]):
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # CNN Dust3R -> RGB features
        self.in_planes = 2048  # for CNN features
        cnn_block_chans = [1024, 512, 256, 128]
        cnns_adaptor_layers = []
        for input_block_chan in cnn_block_chans:
            cnns_adaptor_layers.append(self._make_adaptor_layer(
                input_block_chan, stride=1, norm_layer=nn.InstanceNorm2d,
                upscale = True if input_block_chan != cnn_block_chans[-1] else False)
            )
        self.cnns_adaptor = nn.Sequential(*cnns_adaptor_layers)

        # Transformer Dust3R -> RGB features
        self.in_planes = 512  # for CNN features
        trans_block_chans = [256, 128]
        trans_adaptor_layers = []
        for input_block_chan in trans_block_chans:
            trans_adaptor_layers.append(self._make_adaptor_layer(
                input_block_chan, stride=1, norm_layer=nn.InstanceNorm2d,
                upscale = True if input_block_chan != trans_block_chans[-1] else False)
            )
        self.trans_adaptor = nn.Sequential(*trans_adaptor_layers)

        fused_channel = cnn_block_chans[-1] + cnn_block_chans[-1]
        self.upsampler = nn.Sequential(
            nn.Conv2d(fused_channel, fused_channel, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.GELU(),
        )

        self.num_channels = 1
        feature_dim = 256
        last_dim = 32 
        self.density_head = nn.Sequential(
                nn.Conv2d(32, 32 // 2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32 // 2, last_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0)
            )

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        depth_unet_feat_dim = cfg.depth_unet_feat_dim
        input_channels = 3 + depth_unet_feat_dim + 1
        channels = 32
        depth_unet_channel_mult=cfg.depth_unet_channel_mult
        depth_unet_attn_res=cfg.depth_unet_attn_res,
        self.refine_unet = nn.Sequential(
                nn.Conv2d(42, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                ),
            )
        self.proj_feature = nn.Conv2d(
            fused_channel, 32, 3, 1, 1
        )
        # self.high_resolution_skip = nn.Sequential(
        #     nn.Conv2d(3, cfg.d_feature, 7, 1, 3),
        #     nn.ReLU(),
        # )
        feature_channels = 32
        gau_in = depth_unet_feat_dim + 3
        gaussian_raw_channels = 84
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

    def _make_adaptor_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d, upscale=True):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        if upscale == True:
            uplayer = Upsample(dim, use_conv=True, dims=2)
            layers = (layer1, layer2, uplayer)
        else:
            layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        trans_features,
        cnns_features,
        poses_rel,
        depths,
        depth_conf,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        cnns_features = self.cnns_adaptor(cnns_features)
        trans_features = self.trans_adaptor(trans_features)     #卷积降dim维度  插值h w
        trans_features = F.interpolate(trans_features, scale_factor=1/2, mode="bilinear")

        concat_features = self.upsampler(torch.cat((trans_features,cnns_features),dim=1))
        proj_feature = self.proj_feature(concat_features)

        #################################### density ################################
        context["image"] = rearrange(context["image"], "b v c h w -> (b v) c h w")
        plucker_emb = plucker_embedding(h, w, context["intrinsics"][0], poses_rel[0], jitter=False)
        refine_out = self.refine_unet(torch.cat([context["image"],proj_feature,  \
                    rearrange(depths.unsqueeze(-1), "b v h w l -> (b v) l h w"), plucker_emb],dim=1))
        
        densities = self.density_head(refine_out)
        densities = rearrange(densities.unsqueeze(-1), "(b v) l h w k  -> b v (h w) k l", b =b, v=v)
        
        raw_gaussians = self.to_gaussians(torch.cat([refine_out, context["image"]],dim=1))
        raw_gaussians = rearrange(
                raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
            )
        gaussians = rearrange(raw_gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
        
        depths = rearrange(depths.unsqueeze(-1).unsqueeze(-1), "b v h w k l -> b v (h w) k l")

        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(poses_rel, "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c",),
            (h, w),
        )
        
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
