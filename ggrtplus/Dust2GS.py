import os
import math
import torch

from ggrtplus.checkpoint_manager import CheckPointManager
from ggrtplus.mvsplat.decoder import get_decoder
from ggrtplus.mvsplat.encoder import get_encoder
from ggrtplus.mvsplat.dustsplat import dustSplat

from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.image import resize_dust, rgb
from dust3r.image_pairs import make_pairs
from dust3r.model import AsymmetricCroCo3DStereo, inf 
from dust3r.losses import *

class Dust2GSModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        output_path = os.path.join(self.args.rootdir, "out", self.args.expname)
        if self.args.local_rank == 0:
            os.makedirs(output_path, exist_ok=True)
            print(f'[INFO] Outputs will be saved to {output_path}')
        self.ckpt_manager = CheckPointManager(
            save_path=output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5
        )

        device = torch.device(f'cuda:{args.local_rank}')
        self.dust3r = AsymmetricCroCo3DStereo(pos_embed='RoPE100', 
                    patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), 
                    head_type='dpt', output_mode='pts3d', 
                    depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), 
                    enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, 
                    dec_embed_dim=768, dec_depth=12, dec_num_heads=12).to(device)
        self.dust3r_criterion = ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2).to(device)

        encoder, encoder_visualizer = get_encoder(args.mvsplat.encoder)
        decoder = get_decoder(args.mvsplat.decoder)
        self.gaussian_model = dustSplat(encoder, decoder, encoder_visualizer).to(device)

        self.setup_optimizer()
        self.compose_state_dicts()
        self.start_step = self.load_checkpoint(load_optimizer=load_opt, load_scheduler=load_scheduler)
        if args.distributed:
            self.to_distributed()

    def setup_optimizer(self):
        self.gs_optimizer = torch.optim.Adam(self.gaussian_model.parameters(), 
                                          lr=self.args.optimizer.lr)
        warm_up_steps = self.args.optimizer.warm_up_steps
        self.gs_scheduler = torch.optim.lr_scheduler.LinearLR(self.gs_optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)

        self.dust3r_optimizer = torch.optim.Adam([
            dict(params=self.dust3r.parameters(), lr=self.args.optimizer.lr_dust3r)
        ])
        self.dust3r_scheduler = torch.optim.lr_scheduler.StepLR(
            self.dust3r_optimizer, step_size=self.args.optimizer.lr_dust3r_decay_steps, gamma=0.5)

    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict()}
        self.state_dicts['models']['dust3r'] = self.dust3r
        self.state_dicts['models']['gaussian'] = self.gaussian_model
        
        self.state_dicts['optimizers']['gaussian_optimizer'] = self.gs_optimizer
        self.state_dicts['optimizers']['gaussian_optimizer'] = self.gs_scheduler
        self.state_dicts['optimizers']['dust3r_optimizer'] = self.dust3r_optimizer
        self.state_dicts['schedulers']['dust3r_scheduler'] = self.dust3r_scheduler

    def to_distributed(self):
        self.dust3r = torch.nn.parallel.DistributedDataParallel(
            self.dust3r,
            device_ids=[self.args.local_rank],
            output_device=[self.args.local_rank],
            find_unused_parameters=True
        )
        self.gaussian_model = torch.nn.parallel.DistributedDataParallel(
            self.gaussian_model,
            device_ids=[self.args.local_rank],
            output_device=[self.args.local_rank],
            # find_unused_parameters=True
        )
    def save_checkpoint(self, score: float = 0.0, step = 0) -> None:
        assert self.state_dicts is not None

        self.ckpt_manager.save(
            models=self.state_dicts['models'],
            optimizers=self.state_dicts['optimizers'],
            schedulers=self.state_dicts['schedulers'],
            step=step,
            score=score
        )

    def load_checkpoint(self, load_optimizer=True, load_scheduler=True) -> int:
        iter_start = self.ckpt_manager.load(
            config=self.args,
            models=self.state_dicts['models'],
            optimizers=self.state_dicts['optimizers'] if load_optimizer else None,
            schedulers=self.state_dicts['schedulers'] if load_scheduler else None
        )

        return iter_start

    def switch_to_eval(self):
        self.dust3r.eval()
        self.gaussian_model.eval()

    def switch_to_train(self):
        self.dust3r.train(True)
        self.gaussian_model.train()
            
    def check_if_same_size(self ,pairs):
        shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
        shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
        return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


    def _interleave_imgs(self,img1, img2):
        res = {}
        for key, value1 in img1.items():
            value2 = img2[key]
            if isinstance(value1, torch.Tensor):
                value = torch.stack((value1, value2), dim=1).flatten(0, 1)
            else:
                value = [x for pair in zip(value1, value2) for x in pair]
            res[key] = value
        return res

    def make_batch_symmetric(self , batch):
        view1, view2 = batch
        view1, view2 = (self._interleave_imgs(view1, view2), self._interleave_imgs(view2, view1))
        return view1, view2

    def loss_of_one_batch(self,batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
        view1, view2 = batch
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        if symmetrize_batch:
            view1, view2 = self.make_batch_symmetric(batch)

        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            pred1, pred2 ,feat1, feat2 , path_1 ,path_2= model(view1, view2)

            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                # loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None
                loss=0
        result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
        return  result,feat1,feat2,path_1 ,path_2

    def correct_poses(self, batch,device,batch_size,silent):
        """
        Args:
            fmaps: [n_views+1, c, h, w]
            target_image: [1, h, w, 3]
            ref_imgs: [1, n_views, h, w, 3]
            target_camera: [1, 34]
            ref_cameras: [1, n_views, 34]
        Return:
            inv_depths: n_iters*[1, 1, h, w] if training else [1, 1, h, w]
            rel_poses: [n_views, n_iters, 6] if training else [n_views, 6]
        """
        imgs = resize_dust(batch["context"]["dust_img"],size=512)  
        pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
        # if verbose:
        #     print(f'>> Inference with model on {len(pairs)} image pairs')
        result = []

    # first, check if all images have the same size
        multiple_shapes = not (self.check_if_same_size(pairs))
        if multiple_shapes:  # force bs=1
            batch_size = 1
    # batch_size = len(pairs)
        batch_size = 1
        feat1_list = []
        feat2_list = []
        cnn1_list = []
        cnn2_list = []
        for i in range(0, len(pairs), batch_size):
            view1_ft_lst = []
            view2_ft_lst = []
            # start = time.time()
            loss_tuple,cnn1 ,cnn2,path_1 ,path_2 =  self.loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), self.dust3r, self.dust3r_criterion, device,
                                        symmetrize_batch=False,
                                        use_amp=False, ret='loss')
            # res ,cnn1 ,cnn2,path_1 ,path_2= loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), model, None, device)
            # end =time.time()
            # print(end-start)
            result.append(to_cpu(loss_tuple))
            feat1 = path_1
            feat2 = path_2
            feat1_list.append(feat1)
            feat2_list.append(feat2)
            cnn1_list.append(cnn1)
            cnn2_list.append(cnn2)
        # pfeat01.append(dec2[0])
        result = collate_with_cat(result, lists=multiple_shapes)

        return result,feat1_list,feat2_list,cnn1_list,cnn2_list,imgs

    def switch_state_machine(self, state='joint') -> str:
        if state == 'gs_only':
            self._set_dust3r_state(opt=False)
            self._set_gaussian_state(opt=True)
        
        elif state == 'joint':
            self._set_dust3r_state(opt=True)
            self._set_gaussian_state(opt=True)
        
        else:
            raise NotImplementedError("Not supported state")
        
        return state

    def _set_dust3r_state(self, opt=True):
        for param in self.dust3r.parameters():
            param.requires_grad = opt

    def _set_gaussian_state(self, opt=True):
        for param in self.gaussian_model.parameters():
            param.requires_grad = opt
    
    def compose_joint_loss(self, sfm_loss, nerf_loss, step, coefficient=1e-5):
        # The jointly training loss is composed by the convex_combination:
        #   L = a * L1 + (1-a) * L2
        alpha = math.pow(2.0, -coefficient * step)
        loss = alpha * sfm_loss + (1 - alpha) * nerf_loss
        
        return loss
