import os
import time
import wandb
import config
import imageio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as pl
from einops import rearrange

import torch
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import img2mse, mse2psnr, cycle, data_shim, depth_map, load_config
from ggrtplus.pose_util import Pose, rotation_distance
from ggrtplus.global_cfg import set_cfg
from ggrtplus.data_loaders import dataset_dict
from ggrtplus.GGRtPlus import GGRtPlusModel
from ggrtplus.criterion import MaskedL2ImageLoss, MultiViewPhotometricDecayLoss
from ggrtplus.data_loaders.create_training_dataset import create_training_dataset
from ggrtplus.data_loaders.geometryutils import relative_transformation
from ggrtplus.geometry.align_poses import align_ate_c2b_use_a2b
from ggrtplus.visualization.pose_visualizer import visualize_cameras

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from dust3r_demo import _convert_scene_output_to_glb

from ggrtplus.mvsplat.ply_export import export_ply


def normalize_intrinsics(intrinsics, img_size):
    h, w = img_size
    # 归一化内参矩阵
    intrinsics_normalized = intrinsics.clone()
    intrinsics_normalized[:, 0, 0] /= w
    intrinsics_normalized[:, 1, 1] /= h
    intrinsics_normalized[:, 0, 2] = 0.5
    intrinsics_normalized[:, 1, 2] = 0.5
    return intrinsics_normalized

def eval(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.yaml")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} : {}\n".format(arg, attr))

    # create validation dataset
    # val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
    val_dataset = dataset_dict[args.eval_dataset](args, "train", scenes=args.train_scenes)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create MvSplat model
    model = GGRtPlusModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )

    # Create criterion
    rgb_loss = MaskedL2ImageLoss()
    silent=args.silent

    epoch, global_step = 0, 1
    while global_step < args.n_iters + 1:
        np.random.seed()
        for batch in val_loader:
            scalars_to_log = {}
            time0 = time.time()
            if global_step == 1:
                state = model.switch_state_machine(state='gs_only')

            # output, feat1, feat2, cnn1, cnn2, imgs = model.correct_poses(batch['dust_img'], device, 1)
            # mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
            # scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
            # lr = 0.01
            # if mode == GlobalAlignerMode.PointCloudOptimizer:
            #     loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=lr)

            # depths = scene.get_depthmaps()
            # depths = torch.stack([d.unsqueeze(0) for d in depths], dim=0)
            # images = scene.imgs
            # poses_est = scene.get_im_poses().detach()
            # intrinsic_est = scene.get_intrinsics().detach()
            # intrinsic_est = normalize_intrinsics(intrinsic_est, images[0].shape[-3:-1])
            # # ########################################  scale up extrinsics  #########################################
            # a, b = poses_est[1:3, :3, 3]
            # scale =(a-b).norm()
            # poses_est[:, :3, 3] /= scale
            # poses_est[:, :3, 3] *= 100

            if False:
                min_conf_thr = 1
                rgbimg = scene.imgs
                focals = scene.get_focals().cpu()
                cams2world = scene.get_im_poses().cpu()
                # 3D pointcloud from depthmap, poses and intrinsics
                pts3d = to_numpy(scene.get_pts3d())
                scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
                mask = to_numpy(scene.get_masks())

                _convert_scene_output_to_glb("out", rgbimg, pts3d, mask, focals, cams2world, as_pointcloud=True,
                                        transparent_cams=False, cam_size=0.05, silent=silent) 

            # near, far = torch.tensor(0.1), torch.tensor(100)
            # batch['context'], batch['target'] = {}, {}
            # batch['context']['near'] = depths.min().repeat(2)[None, ...] / scale
            # batch['context']['far'] = depths.max().repeat(2)[None, ...] / scale
            # batch['context']['extrinsics'] = torch.cat([poses_est[0][None, ...], poses_est[2][None, ...]]).unsqueeze(0)
            # batch['context']['intrinsics'] = torch.cat([intrinsic_est[0][None, ...], intrinsic_est[2][None, ...]]).unsqueeze(0)
            # batch['context']['image'] = torch.from_numpy(np.stack([images[0], images[2]])).to(poses_est.device).permute(0,3,1,2).unsqueeze(0)

            # batch['target']['near'] = depths.min()[None, ...][None, ...] / scale
            # batch['target']['far'] = depths.max()[None, ...][None, ...] / scale
            # batch['target']['extrinsics'] = poses_est[1:2].unsqueeze(0)
            # batch['target']['intrinsics'] = intrinsic_est[1:2].unsqueeze(0)
            # batch['target']['image'] = torch.from_numpy(np.stack(images[1:2])).to(poses_est.device).permute(0,3,1,2).unsqueeze(0)
            batch = data_shim(batch, device=device)

            ret, data_gt, ret_ref, data_gt_ref = model.gaussian_model(batch, global_step)

            if False:
                export_ply(
                    batch["context"]["extrinsics"][0, 0],
                    gaussians.means[0],
                    visualization_dump["scales"][0],
                    visualization_dump["rotations"][0],
                    gaussians.harmonics[0],
                    gaussians.opacities[0],
                    Path(os.path.join("out", f"000.ply")),
                )

            # compute loss
            model.gs_optimizer.zero_grad()
            model.dust3r_optimizer.zero_grad()
            coarse_loss = rgb_loss(ret, data_gt)
            loss_all = coarse_loss
            if args.use_aux_loss == True:
                loss_all += 0.1 * rgb_loss(ret_ref, data_gt_ref)

            loss_all.backward()
            model.gs_optimizer.step()
            model.gs_scheduler.step()
            model.dust3r_optimizer.step()
            model.dust3r_scheduler.step()

            lr = model.gs_scheduler.get_last_lr()[0]
            scalars_to_log["train/rgb-loss"] = coarse_loss
            
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.n_logging == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(data_gt["rgb"], ret["rgb"]).item()
                    scalars_to_log["train/psnr"] = mse2psnr(mse_error)
                    logstr = "{} Epoch: {} Step: {} Lr: {:.05f}".format(args.expname, epoch, global_step, lr)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.3f}".format(k, scalars_to_log[k])
                    print(logstr, "| {:.02f} s/iter".format(dt))
                    
                    scalars_to_log["train/step"] = global_step
                    scalars_to_log["lr"] = lr
                    # if args.expname != 'debug':
                    #     render_rgb = (255 * np.clip(ret['rgb'][0][0].permute(1,2,0).detach().cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                    #     render_depth = depth_map(ret['depth'][0][0] + 1e-7)
                    #     est_depth = depth_map(depths[-1, ...])
                    #     wandb.log({
                    #         "Image": {
                    #             'rendered': wandb.Image(render_rgb, caption='Rendered Image'),
                    #             'gt': wandb.Image(data_gt['rgb'][0][0].permute(1,2,0).detach().cpu().numpy(), caption='GT Image')},
                    #         "Depth": {'rendered': wandb.Image(render_depth.permute(1,2,0).detach().cpu().numpy(), caption='Rendered Depth'),
                    #                 'est': wandb.Image(est_depth[0].permute(1,2,0).detach().cpu().numpy(), caption='Est Depth')}
                    #             })
                    #     wandb.log(scalars_to_log)
                if (global_step+1) % args.n_checkpoint == 0:
                    print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    model.save_checkpoint(score=0, step=global_step)

                # if global_step % args.n_validation == 0:
                #     print("Logging current training view...")
                #     log_view(
                #         global_step,
                #         args,
                #         model,
                #         render_stride=1,
                #         prefix="train/",
                #         out_folder=out_folder,
                #         ret_alpha=args.N_importance > 0,
                #         single_net=args.single_net,
                #     )
                global_step += 1
                if global_step > model.start_step + args.n_iters + 1:
                    break
        epoch += 1

if __name__ == "__main__":
    parser = config.config_parser()
    args_override = parser.parse_args()
    args = load_config(args_override.config)
    defaults = {action.dest: action.default for action in parser._actions}
    for arg_override, value in vars(args_override).items():
        if arg_override in defaults and value != defaults[arg_override] and arg_override != "config":
            args.__dict__[arg_override] = value
    set_cfg(args)

    eval(args)
