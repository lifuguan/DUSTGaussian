import os
import time
import wandb
import config
import imageio
import visdom
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
from ggrtplus.Dust2GS import Dust2GSModel
from ggrtplus.criterion import MaskedL2ImageLoss, MultiViewPhotometricDecayLoss
from ggrtplus.data_loaders.create_training_dataset import create_training_dataset
from ggrtplus.data_loaders.geometryutils import relative_transformation
from ggrtplus.geometry.align_poses import align_ate_c2b_use_a2b
from ggrtplus.visualization.pose_visualizer import visualize_cameras

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

from dust3r_demo import _convert_scene_output_to_glb


@torch.no_grad()
def get_predicted_training_poses(pred_poses):
    target_pose = torch.eye(4, device=pred_poses.device, dtype=torch.float).repeat(1, 1, 1)

    # World->camera poses.
    pred_poses = Pose.from_vec(pred_poses) # [n_views, 4, 4]
    pred_poses = torch.cat([target_pose, pred_poses], dim=0)

    # Convert camera poses to camera->world.
    pred_poses = pred_poses.inverse()

    return pred_poses


@torch.no_grad()
def align_predicted_training_poses(pred_poses, poses_gt, device='cpu'):
    aligned_pred_poses = align_ate_c2b_use_a2b(pred_poses, poses_gt)
    return aligned_pred_poses, poses_gt

@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    # measure errors in rotation and translation
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)
    
    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
    
    mean_rotation_error = np.rad2deg(R_error.mean().cpu())
    mean_position_error = t_error.mean()
    med_rotation_error = np.rad2deg(R_error.median().cpu())
    med_position_error = t_error.median()
    
    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.local_rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.local_rank=0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.local_rank)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def train(args):

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

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create MvSplat model
    model = Dust2GSModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler
    )

    # Create criterion
    rgb_loss = MaskedL2ImageLoss()
    photometric_loss = MultiViewPhotometricDecayLoss()
    scalars_to_log = {}

    silent=args.silent
    # visdom_ins = visdom.Visdom(server='localhost', port=8097, env='splatam')

    epoch = 0
    # global_step = model.start_step + 1
    # while global_step < model.start_step + args.n_iters + 1:
    global_step = 1
    while global_step < args.n_iters + 1:
        np.random.seed()
        for batch in train_loader:
            scene_name = batch['scene_name']
            time0 = time.time()
            if global_step == 1:
                state = model.switch_state_machine(state='gs_only')

            if args.distributed:
                train_sampler.set_epoch(epoch)

            output, feat1, feat2, cnn1, cnn2, imgs = model.correct_poses(batch['dust_img'], device, 1)
            mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
            lr = 0.01
            if mode == GlobalAlignerMode.PointCloudOptimizer:
                loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=lr)

            depths = scene.get_depthmaps()
            depths = torch.stack([d.unsqueeze(0) for d in depths], dim=0)
            poses_est = scene.get_im_poses().detach()
            poses_gt = torch.cat([batch['context']["extrinsics"][0], batch['target']["extrinsics"][0]])
            confs = [to_numpy(c) for c in scene.im_conf]    #在每个视角下 点云的置信度
            confs_max = max(c.max() for c in confs)
            confs = torch.stack([torch.from_numpy(pl.get_cmap('jet')(d/confs_max)).permute(2, 0, 1).unsqueeze(0) for d in confs], dim=0)

            ########################################  scale up extrinsics  #########################################
            a, b = poses_est[0:2, :3, 3]
            scale =(a-b).norm()
            poses_est[:, :3, 3] /= scale
            depths /= scale
            ########################################  export GLB file      #########################################
            if False:
                min_conf_thr = 1
                rgbimg = scene.imgs
                focals = scene.get_focals().cpu()
                cams2world = poses_est.cpu()
                # 3D pointcloud from depthmap, poses and intrinsics
                pts3d = to_numpy([pts / scale for pts in scene.get_pts3d()]) 
                scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
                mask = to_numpy(scene.get_masks())

                _convert_scene_output_to_glb("out", rgbimg, pts3d, mask, focals, cams2world, as_pointcloud=True,
                                        transparent_cams=False, cam_size=0.05, silent=silent)    
            
            ######################################  data shim for DustSplat  #########################################
            batch = data_shim(batch, device=device)

            # 批量操作
            confs = confs.mean(dim=0)
            depths = depths.view(-1, *depths.shape[2:])

            # 重构循环，减少重复代码
            def append_to_batch_lists(item, list_name, start, end):
                if list_name not in batch.keys():
                    batch[list_name] = item[start:end].unsqueeze(0)
                else:
                    batch[list_name] = torch.cat([batch[list_name], item[start:end].unsqueeze(0)], dim=1)

            # 使用函数减少重复代码
            num_imgs = len(imgs) - 2
            for i in range(num_imgs):
                append_to_batch_lists(torch.cat([cnn1[i], cnn2[i]], dim=0), 'cnn', i, i+2)
                append_to_batch_lists(torch.cat([feat1[i], feat2[i]], dim=0), 'features', i, i+2)
                append_to_batch_lists(confs[:,i:i+2,:,:], 'confs', i, i+2)
                append_to_batch_lists(depths[i:i+2], 'depths', i, i+2)
                append_to_batch_lists(poses_est[i:i+2], 'pose', i, i+2)
            
            _,_,_,H,W = batch["context"]['image'].shape
            batch['cnn'] = batch['cnn'].permute(0, 1, 3, 2)
            batch['cnn'] = rearrange(batch['cnn'], "b v d (h w) -> b v d h w",h=H//16,w=W//16)
            # batch['target']['extrinsics'] = poses_est[-1:].unsqueeze(0)
            # batch['context']['extrinsics'] = batch['pose']

            ret, data_gt, _, _ = model.gaussian_model(batch, batch['features'], batch['cnn'], \
                batch['depths'], batch['confs'].float(), global_step)

            # compute loss
            model.gs_optimizer.zero_grad()
            if state != 'gs_only':
                model.dust3r_optimizer.zero_grad()
            coarse_loss = rgb_loss(ret, data_gt)
            loss_all = coarse_loss

            loss_all.backward()
            model.gs_optimizer.step()
            model.gs_scheduler.step()
            if state != 'gs_only':
                model.dust3r_optimizer.step()
                model.dust3r_scheduler.step()

            scalars_to_log["train/step"] = global_step
            scalars_to_log["lr"] = model.gs_scheduler.get_last_lr()[0]
            scalars_to_log["train/rgb-loss"] = coarse_loss
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.n_logging == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(data_gt["rgb"], ret["rgb"]).item()
                    scalars_to_log["train/psnr"] = mse2psnr(mse_error)

                    pose_error = evaluate_camera_alignment(poses_est, poses_gt.to(poses_est.device))  
                    scalars_to_log["train/R_err"] = pose_error['R_error_mean']
                    scalars_to_log["train/t_err"] = pose_error['t_error_mean']
                    # visualize_cameras(visdom_ins, step=global_step, poses=[poses_est, poses_gt], cam_depth=0.1, caption="not aligned")

                    logstr = "{} Epoch: {} Scene: {} ".format(args.expname, epoch, scene_name)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.3f}".format(k, scalars_to_log[k])
                    print(logstr, " | {:.02f} s/iter".format(dt))
                    
                    if args.expname != 'debug':
                        render_rgb = (255 * np.clip(ret['rgb'][0][0].permute(1,2,0).detach().cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                        render_depth = depth_map(ret['depth'][0][0] + 1e-7)
                        est_depth = depth_map(depths[-1, ...])
                        wandb.log({
                            "Image": {
                                'rendered': wandb.Image(render_rgb, caption='Rendered Image'),
                                'gt': wandb.Image(data_gt['rgb'][0][0].permute(1,2,0).detach().cpu().numpy(), caption='GT Image')},
                            "Depth": {'rendered': wandb.Image(render_depth.permute(1,2,0).detach().cpu().numpy(), caption='Rendered Depth'),
                                    'est': wandb.Image(est_depth.permute(1,2,0).detach().cpu().numpy(), caption='Est Depth')}
                                })
                        wandb.log(scalars_to_log)
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
    init_distributed_mode(args)
    if args.local_rank == 0 and args.expname != 'debug':
        wandb.init(
            # set the wandb project where this run will be logged
            entity="vio-research",
            project="dust2gs",
            name=args.expname,
            config=args
        )
        wandb.define_metric("train/step")
    train(args)
