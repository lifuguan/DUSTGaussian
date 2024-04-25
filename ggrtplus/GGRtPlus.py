import os
import math
import torch

from ggrtplus.checkpoint_manager import CheckPointManager
from ggrtplus.mvsplat.decoder import get_decoder
from ggrtplus.mvsplat.encoder import get_encoder
from ggrtplus.mvsplat.mvsplat import MvSplat

class MvSplatModel(object):
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

        # create generalized 3d gaussian.
        encoder, encoder_visualizer = get_encoder(args.mvsplat.encoder)
        decoder = get_decoder(args.mvsplat.decoder)
        self.gaussian_model = MvSplat(encoder, decoder, encoder_visualizer).to(torch.device(f'cuda:{args.local_rank}'))
        
        self.setup_optimizer()
        self.start_step = self.load_checkpoint(load_optimizer=load_opt, load_scheduler=load_scheduler)
        if args.distributed:
            self.to_distributed()

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.gaussian_model.parameters(), 
            lr=self.args.optimizer.lr)
        warm_up_steps = self.args.optimizer.warm_up_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)
        self.state_dicts = {
            'models': self.gaussian_model, 
            'optimizers':  self.optimizer,
            'schedulers': self.scheduler}


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

    def to_distributed(self):
        self.gaussian_model = torch.nn.parallel.DistributedDataParallel(
            self.gaussian_model,
            device_ids=[self.args.local_rank],
            output_device=[self.args.local_rank]
        )


    def switch_to_eval(self):
        self.gaussian_model.eval()

    def switch_to_train(self):
        self.gaussian_model.train()
     
    def switch_state_machine(self, state='nerf_only') -> str:
        if state == 'nerf_only':
            self._set_gaussian_state(opt=True)
        else:
            raise NotImplementedError("Not supported state")
        
        return state
    def _set_gaussian_state(self, opt=True):
        for param in self.gaussian_model.parameters():
            param.requires_grad = opt