"""
Self-supervised DINO training engine.
"""

import math
import random
from datetime import datetime
import logging

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch

from watchmal.engine.reconstruction import ReconstructionEngine

log = logging.getLogger(__name__)


class DINOViewGenerator:
    """
    Generate weak teacher views and strongly corrupted student views.
    """

    def __init__(
        self,
        time_channel=0,
        charge_channel=1,
        hit_threshold=0.0,
        teacher_hit_dropout=(0.0, 0.02),
        teacher_time_jitter=(0.0, 0.2),
        teacher_charge_jitter=(0.0, 0.01),
        student_hit_dropout=(0.0, 0.05),
        student_time_jitter=(0.5, 1.5),
        student_charge_jitter=(0.02, 0.08),
        student_mask_fraction=(0.5, 0.8),
        student_block_sizes=(12, 16, 24, 32),
        student_max_blocks=12,
        student_max_attempts=50,
        min_hits_for_block_mask=100,
        dark_noise_mean=0.0,
        dark_noise_charge=(0.02, 0.2),
    ):
        self.time_channel = time_channel
        self.charge_channel = charge_channel
        self.hit_threshold = hit_threshold
        self.teacher_hit_dropout = tuple(teacher_hit_dropout)
        self.teacher_time_jitter = tuple(teacher_time_jitter)
        self.teacher_charge_jitter = tuple(teacher_charge_jitter)
        self.student_hit_dropout = tuple(student_hit_dropout)
        self.student_time_jitter = tuple(student_time_jitter)
        self.student_charge_jitter = tuple(student_charge_jitter)
        self.student_mask_fraction = tuple(student_mask_fraction)
        self.student_block_sizes = tuple(student_block_sizes)
        self.student_max_blocks = int(student_max_blocks)
        self.student_max_attempts = int(student_max_attempts)
        self.min_hits_for_block_mask = int(min_hits_for_block_mask)
        self.dark_noise_mean = float(dark_noise_mean)
        self.dark_noise_charge = tuple(dark_noise_charge)

    def teacher_view(self, x):
        view = x.clone()
        self._hit_dropout(view, self.teacher_hit_dropout)
        self._jitter(view, self.teacher_time_jitter, self.teacher_charge_jitter)
        return view

    def student_view(self, x):
        view = x.clone()
        stats = self._local_block_mask(view)
        self._hit_dropout(view, self.student_hit_dropout)
        self._add_dark_noise(view)
        self._jitter(view, self.student_time_jitter, self.student_charge_jitter)
        return view, stats

    def _uniform(self, bounds):
        low, high = bounds
        if high <= low:
            return float(low)
        return random.uniform(float(low), float(high))

    def _hit_mask(self, x):
        return x[:, self.charge_channel] > self.hit_threshold

    def _hit_dropout(self, x, dropout_range):
        probability = self._uniform(dropout_range)
        if probability <= 0:
            return
        hit_mask = self._hit_mask(x)
        drop_mask = torch.rand_like(hit_mask.float()) < probability
        drop_mask = drop_mask & hit_mask
        x.masked_fill_(drop_mask.unsqueeze(1), 0.0)

    def _jitter(self, x, time_sigma_range, charge_sigma_range):
        hit_mask = self._hit_mask(x)
        if not hit_mask.any():
            return
        time_sigma = self._uniform(time_sigma_range)
        charge_sigma = self._uniform(charge_sigma_range)
        if time_sigma > 0:
            noise = torch.randn_like(x[:, self.time_channel]) * time_sigma
            x[:, self.time_channel] = torch.where(hit_mask, x[:, self.time_channel] + noise, x[:, self.time_channel])
        if charge_sigma > 0:
            noise = torch.randn_like(x[:, self.charge_channel]) * charge_sigma
            scale = torch.exp(noise).to(dtype=x.dtype)
            x[:, self.charge_channel] = torch.where(hit_mask, x[:, self.charge_channel] * scale, x[:, self.charge_channel])
            x[:, self.charge_channel].clamp_(min=0.0)

    def _local_block_mask(self, x):
        B, _, H, W = x.shape
        hit_mask = self._hit_mask(x)
        masked_fractions = []
        masked_charge_fractions = []
        for batch_index in range(B):
            event_hits = hit_mask[batch_index]
            n_hits = int(event_hits.sum().item())
            total_charge = x[batch_index, self.charge_channel][event_hits].sum().clamp_min(1e-8)
            if n_hits < self.min_hits_for_block_mask:
                masked_fractions.append(x.new_tensor(0.0))
                masked_charge_fractions.append(x.new_tensor(0.0))
                continue

            target_fraction = self._uniform(self.student_mask_fraction)
            target_hits = max(1, int(math.ceil(target_fraction * n_hits)))
            mask = torch.zeros((H, W), dtype=torch.bool, device=x.device)
            blocks_used = 0
            attempts = 0
            while int((mask & event_hits).sum().item()) < target_hits and attempts < self.student_max_attempts:
                attempts += 1
                available_hits = event_hits & (~mask)
                if not available_hits.any():
                    break
                coords = torch.nonzero(available_hits, as_tuple=False)
                center = coords[torch.randint(coords.shape[0], (1,), device=x.device).item()]
                block_size = random.choice(self.student_block_sizes)
                half = block_size // 2
                row = int(center[0].item())
                col = int(center[1].item())
                r0 = max(0, row - half)
                r1 = min(H, row + half + (block_size % 2))
                c0 = max(0, col - half)
                c1 = min(W, col + half + (block_size % 2))
                candidate = torch.zeros_like(mask)
                candidate[r0:r1, c0:c1] = True
                if int((candidate & available_hits).sum().item()) == 0:
                    continue
                mask |= candidate
                blocks_used += 1
                if blocks_used >= self.student_max_blocks:
                    break

            masked_hits = int((mask & event_hits).sum().item())
            if masked_hits < target_hits:
                needed = target_hits - masked_hits
                available_hits = event_hits & (~mask)
                coords = torch.nonzero(available_hits, as_tuple=False)
                if coords.shape[0] > 0:
                    permutation = torch.randperm(coords.shape[0], device=x.device)[:needed]
                    selected = coords[permutation]
                    mask[selected[:, 0], selected[:, 1]] = True

            final_mask = mask & event_hits
            masked_charge = x[batch_index, self.charge_channel][final_mask].sum()
            x[batch_index, :, final_mask] = 0.0
            masked_fractions.append(x.new_tensor(float(final_mask.sum().item()) / max(n_hits, 1)))
            masked_charge_fractions.append((masked_charge / total_charge).detach())

        return {
            "masked_hit_fraction": torch.stack(masked_fractions).mean(),
            "masked_charge_fraction": torch.stack(masked_charge_fractions).mean(),
        }

    def _add_dark_noise(self, x):
        if self.dark_noise_mean <= 0:
            return
        B, _, H, W = x.shape
        for batch_index in range(B):
            n_noise = int(torch.poisson(x.new_tensor(self.dark_noise_mean)).item())
            if n_noise <= 0:
                continue
            hit_mask = x[batch_index, self.charge_channel] > self.hit_threshold
            empty_coords = torch.nonzero(~hit_mask, as_tuple=False)
            if empty_coords.shape[0] == 0:
                continue
            selected = empty_coords[torch.randperm(empty_coords.shape[0], device=x.device)[:n_noise]]
            charge_low, charge_high = self.dark_noise_charge
            charges = torch.empty(selected.shape[0], device=x.device, dtype=x.dtype).uniform_(float(charge_low), float(charge_high))
            existing_times = x[batch_index, self.time_channel][hit_mask]
            if existing_times.numel() > 0:
                t_min = float(existing_times.min().item())
                t_max = float(existing_times.max().item())
            else:
                t_min, t_max = 0.0, 1.0
            times = torch.empty(selected.shape[0], device=x.device, dtype=x.dtype).uniform_(t_min, t_max)
            x[batch_index, self.time_channel, selected[:, 0], selected[:, 1]] = times
            x[batch_index, self.charge_channel, selected[:, 0], selected[:, 1]] = charges


class DINOEngine(ReconstructionEngine):
    """Engine for label-free SwinT-DINO pre-training."""

    def __init__(
        self,
        model,
        rank,
        device,
        dump_path,
        target_key=None,
        teacher_momentum=0.996,
        teacher_momentum_final=None,
        teacher_momentum_steps=None,
        num_teacher_views=2,
        num_student_views=2,
        view_generator=None,
    ):
        super().__init__(target_key, model, rank, device, dump_path)
        self.teacher_momentum = float(teacher_momentum)
        self.teacher_momentum_final = teacher_momentum_final
        self.teacher_momentum_steps = teacher_momentum_steps
        self.num_teacher_views = int(num_teacher_views)
        self.num_student_views = int(num_student_views)
        if view_generator is None:
            self.view_generator = DINOViewGenerator()
        elif isinstance(view_generator, DictConfig):
            self.view_generator = instantiate(view_generator)
        else:
            self.view_generator = view_generator
        self.teacher_outputs = None
        self.student_outputs = None
        self.view_stats = {}
        self.last_momentum = self.teacher_momentum

    def configure_loss(self, loss_config):
        self.criterion = instantiate(loss_config).to(self.device)

    def process_target(self, data):
        return None

    def train(
        self,
        epochs=0,
        val_interval=20,
        num_val_batches=4,
        checkpointing=False,
        save_interval=None,
        max_steps=None,
        resume=False,
    ):
        """
        DINO-specific training loop.
        """
        if self.rank == 0:
            log.info(f"Training {epochs} epochs with {num_val_batches}-batch validation each {val_interval} iterations")
        self.model.train()
        if not resume:
            self.epoch = 0
            self.iteration = 0
            self.best_validation_loss = np.inf
        elif self.rank == 0:
            log.info(f"Resuming DINO training from iteration {self.iteration}, epoch {self.epoch}")
        if self.best_validation_loss is None:
            self.best_validation_loss = np.inf

        val_iter = iter(self.data_loaders["validation"])
        start_time = datetime.now()
        step_time = start_time
        epoch_start_time = start_time
        start_epoch = self.epoch if resume else 0
        for self.epoch in range(start_epoch, epochs):
            if self.rank == 0:
                if self.epoch > 0:
                    log.info(f"Epoch {self.epoch} completed in {datetime.now() - epoch_start_time}")
                    epoch_start_time = datetime.now()
                log.info(f"Epoch {self.epoch+1} starting at {datetime.now()}")

            train_loader = self.data_loaders["train"]
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)
            steps_per_epoch = len(train_loader)
            for step, train_data in enumerate(train_loader):
                if max_steps is not None and step >= max_steps:
                    break
                self.process_data(train_data)
                self.process_target(train_data)
                outputs, metrics = self.step(True, True)
                metrics = {k: v.item() for k, v in metrics.items()}
                self.backward()
                if self.scheduler is not None:
                    self.scheduler.step()
                step += 1
                self.iteration += 1
                log_entries = {"iteration": self.iteration, "epoch": self.epoch, **metrics}
                self.train_log.log(log_entries)
                if self.iteration % val_interval == 0:
                    if self.rank == 0:
                        previous_step_time = step_time
                        step_time = datetime.now()
                        average_step_time = (step_time - previous_step_time) / val_interval
                        print(f"Iteration {self.iteration},"
                              f" Epoch {self.epoch+1}/{epochs},"
                              f" Step {step}/{steps_per_epoch}"
                              f" Step time {average_step_time},"
                              f" Epoch time {step_time-epoch_start_time}"
                              f" Total time {step_time-start_time}")
                        print(f"  Training   {', '.join(f'{k}: {v:.5g}' for k, v in metrics.items())}")
                    self.validate(val_iter, num_val_batches, checkpointing)
            if self.rank == 0 and (save_interval is not None) and ((self.epoch + 1) % save_interval == 0):
                self.save_state(suffix=f'_epoch_{self.epoch+1}')
        self.train_log.close()
        if self.rank == 0:
            log.info(f"Epoch {self.epoch} completed in {datetime.now() - epoch_start_time}")
            log.info(f"Training {epochs} epochs completed in {datetime.now()-start_time}")
            self.val_log.close()

    def forward_pass(self):
        teacher_views = [self.view_generator.teacher_view(self.data) for _ in range(self.num_teacher_views)]
        student_results = [self.view_generator.student_view(self.data) for _ in range(self.num_student_views)]
        student_views = [result[0] for result in student_results]
        stats = [result[1] for result in student_results]
        self.view_stats = {
            key: torch.stack([stat[key] for stat in stats]).mean()
            for key in stats[0].keys()
        } if stats else {}

        self.teacher_outputs = [self.module.forward_teacher(view) for view in teacher_views]
        self.student_outputs = [self.model(view) for view in student_views]
        return {
            "student_logits": torch.cat(self.student_outputs, dim=0),
            "teacher_logits": torch.cat(self.teacher_outputs, dim=0),
        }

    def compute_metrics(self, update_center=True):
        self.loss, loss_details = self.criterion(
            self.student_outputs,
            self.teacher_outputs,
            update_center=update_center,
        )
        metrics = {"loss": self.loss, **loss_details, **self.view_stats}
        if "teacher_momentum" not in metrics:
            metrics["teacher_momentum"] = self.loss.new_tensor(self.last_momentum)
        return metrics

    def step(self, train=True, with_metrics=True):
        with torch.set_grad_enabled(train):
            outputs = self.forward_pass()
            if not with_metrics:
                return outputs
            metrics = self.compute_metrics(update_center=train)
            return outputs, metrics

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.last_momentum = self._current_teacher_momentum()
        self.module.update_teacher(self.last_momentum)

    def _current_teacher_momentum(self):
        if self.teacher_momentum_final is None or self.teacher_momentum_steps is None:
            return self.teacher_momentum
        progress = min(float(self.iteration) / max(float(self.teacher_momentum_steps), 1.0), 1.0)
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return float(self.teacher_momentum_final) - (float(self.teacher_momentum_final) - self.teacher_momentum) * cosine

    def save_state(self, suffix="", name=None):
        """Save DINO model, optimizer, and DINO loss buffers."""
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        filename = f"{self.dump_path}{name}{suffix}.pth"
        state_data = {
            "global_step": self.iteration,
            "epoch": self.epoch,
            "state_dict": self.module.state_dict(),
        }
        if self.best_validation_loss is not None and not np.isinf(self.best_validation_loss):
            state_data["best_validation_loss"] = self.best_validation_loss
        if self.optimizer is not None:
            state_data["optimizer"] = self.optimizer.state_dict()
        if self.criterion is not None:
            state_data["criterion"] = self.criterion.state_dict()
        self.state_data = state_data
        torch.save(self.state_data, filename)
        log.info(f"Saved state as: {filename}")
        return filename

    def restore_state(self, weight_file):
        """Restore DINO state, including loss buffers needed for resume."""
        with open(weight_file, 'rb') as f:
            log.info(f"Restoring state from {weight_file}")
            if self.is_distributed:
                torch.distributed.barrier()
            try:
                self.state_data = torch.load(f, map_location=self.device, weights_only=False)
            except TypeError:
                self.state_data = torch.load(f, map_location=self.device)
            self.module.load_state_dict(self.state_data['state_dict'])
            if self.optimizer is not None and 'optimizer' in self.state_data:
                self.optimizer.load_state_dict(self.state_data['optimizer'])
            if self.criterion is not None and 'criterion' in self.state_data:
                self.criterion.load_state_dict(self.state_data['criterion'])
            self.iteration = self.state_data['global_step']
            self.epoch = self.state_data.get('epoch', 0)
            self.best_validation_loss = self.state_data.get('best_validation_loss', self.best_validation_loss)
