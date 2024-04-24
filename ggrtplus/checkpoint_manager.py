import os
import shutil
import time

import torch
import numpy as np


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class CheckPointManager(object):
    """
    Manager for saving/managing pytorch checkpoints.

    Provides functionality similar to tf.Saver such as
    max_to_keep and keep_checkpoint_every_n_hours
    """
    def __init__(
        self, save_path: str = None, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0
    ):
        if max_to_keep <= 0:
            raise ValueError('max_to_keep must be at least 1')

        self._max_to_keep = max_to_keep
        self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

        self._checkpoints_permanent = []  # Will not be deleted
        self._checkpoints_buffer = []  # Those which might still be deleted
        self._next_save_time = time.time()
        self._best_score = None
        self._best_step = None

        if save_path is not None:
            self._save_path = save_path
            self._checkpoints_fname = os.path.join(save_path, 'checkpoints.txt')
            self._update_checkpoints_file()
        else:
            self._save_path = None
            self._checkpoints_fname = None

    def set_save_path(self, path: str):
        self._save_path = path

    def _save_checkpoint(
        self, step, models, optimizers: dict, schedulers: dict = None, score=0.0
    ):
        checkpoint_dir = os.path.join(self._save_path, 'model')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = os.path.join(checkpoint_dir, 'model_{:06d}'.format(step) + '.pth')

        state_dicts = {'step': step}

        # Append models.
        state_dicts['model'] = models.state_dict()
        
        # Append optimizers.
        state_dicts['optimizer'] = optimizers.state_dict()
        
        # Append schedulers.
        if schedulers != None:
            state_dicts['schedulers'] = schedulers.state_dict()
        
        print(f'Saving checkpoint: {checkpoint_name}')
        torch.save(state_dicts, checkpoint_name)
        
        # Track the latest model.
        shutil.copy(checkpoint_name, os.path.join(self._save_path, 'model.pth'))

        self._checkpoints_buffer.append((checkpoint_name, time.time()))

        if self._best_score is None or np.all(np.array(score) >= np.array(self._best_score)):
            best_checkpoint_name = os.path.join(self._save_path, 'model_best.pth')
            shutil.copyfile(checkpoint_name, best_checkpoint_name)
            self._best_score = score
            self._best_step = step
            print('Checkpoint is current best, score={}'.format(
                np.array_str(np.array(self._best_score), precision=3)))

    def _remove_old_checkpoints(self):
        while len(self._checkpoints_buffer) > self._max_to_keep:
            to_remove = self._checkpoints_buffer.pop(0)

            if to_remove[1] > self._next_save_time:
                self._checkpoints_permanent.append(to_remove)
                self._next_save_time = to_remove[1] + self._keep_checkpoint_every_n_hours * 3600
            else:
                os.remove(to_remove[0])

    def _update_checkpoints_file(self):
        checkpoints = [os.path.basename(c[0]) \
            for c in self._checkpoints_permanent + self._checkpoints_buffer
        ]
        with open(self._checkpoints_fname, 'w') as fid:
            fid.write('\n'.join(checkpoints))
            fid.write('\nBest step: {}'.format(self._best_step))

    def save(self, models: dict, optimizers: dict, step: int, schedulers: dict = None, score: float = 0.0):
        """
        Save model checkpoint to file

        Args:
            model: Torch model
            optimizer: Torch optimizer
            step (int): Step, model will be saved as model-[step].pth
            score (float, optional): To determine which model is the best
        """
        if self._save_path is None:
            raise AssertionError(
                'Checkpoint manager must be initialized with save path for save().')

        self._save_checkpoint(step, models, optimizers, schedulers, score)
        self._remove_old_checkpoints()
        self._update_checkpoints_file()

    def load(self, config, models: dict = None,
             optimizers: dict = None, schedulers: dict = None,):
        """
        Loads saved model from file
        
        Args:
            config: configurations
            models: Torch models to restore weights to
            optimizers: Optimizers
            schedulers: Schedulers
        """
        checkpoint_name = self._save_path
        if os.path.exists(config.ckpt_path):
            # Loaded from the specified checkpoint.
            print(f'[INFO] Resuming from checkpoint {config.ckpt_path}...')
            checkpoint_name = config.ckpt_path
        elif os.path.isdir(self._save_path):
            # Loaded from the latest checkpoint.
            print('[INFO] Resuming from latest checkpoint...')
            checkpoint_name = os.path.join(self._save_path, 'model.pth')
        if not os.path.exists(checkpoint_name):
            print(f"[WARNING] Checkpoint {checkpoint_name} does not exist, training from scratch!")
            return 0

        if config.distributed:
            state = torch.load(checkpoint_name, map_location=f'cuda:{config.local_rank}')
        else:
            # state = torch.load(checkpoint_name)
            state = torch.load(checkpoint_name, map_location='cuda:0')

        step = 0
        if 'step' in state:
            step = state['step']

        if models is not None:
            models.load_state_dict(state['model'])

        if optimizers is not None:
            optimizers.load_state_dict(state['optimizer'])
        
        if schedulers != None:
            schedulers.load_state_dict(state['schedulers'])

        print(f'[INFO] Loaded models from {checkpoint_name}')
        return step
