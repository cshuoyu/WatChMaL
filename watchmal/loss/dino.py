"""
DINO self-distillation loss.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    Cross-entropy between teacher prototype assignments and student prototype
    predictions. The teacher target is produced either by EMA centering or by
    Sinkhorn-Knopp balanced assignment.
    """

    def __init__(
        self,
        out_dim=1024,
        student_temp=0.1,
        teacher_temp=0.04,
        center_momentum=0.9,
        target_method="center",
        sinkhorn_iters=3,
        teacher_temp_final=None,
        teacher_temp_warmup_steps=None,
    ):
        super().__init__()
        if target_method not in ("center", "sinkhorn"):
            raise ValueError(
                f"target_method must be 'center' or 'sinkhorn', got {target_method!r}"
            )
        self.student_temp = float(student_temp)
        self.teacher_temp_init = float(teacher_temp)
        self.teacher_temp_final = (
            float(teacher_temp_final) if teacher_temp_final is not None else None
        )
        self.teacher_temp_warmup_steps = (
            int(teacher_temp_warmup_steps)
            if teacher_temp_warmup_steps is not None
            else None
        )
        self.center_momentum = float(center_momentum)
        self.target_method = target_method
        self.sinkhorn_iters = int(sinkhorn_iters)
        # Center buffer is registered unconditionally so checkpoints stay
        # interchangeable between the two modes.
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @property
    def teacher_temp(self):
        if (
            self.teacher_temp_final is None
            or self.teacher_temp_warmup_steps is None
            or self.teacher_temp_warmup_steps <= 0
        ):
            return self.teacher_temp_init
        progress = min(float(self.step.item()) / float(self.teacher_temp_warmup_steps), 1.0)
        return self.teacher_temp_init + (self.teacher_temp_final - self.teacher_temp_init) * progress

    def forward(self, student_outputs, teacher_outputs, update_center=True):
        if not isinstance(student_outputs, (list, tuple)):
            student_outputs = [student_outputs]
        if not isinstance(teacher_outputs, (list, tuple)):
            teacher_outputs = [teacher_outputs]

        teacher_temp = self.teacher_temp

        student_log_probs = [
            F.log_softmax(student_output / self.student_temp, dim=-1)
            for student_output in student_outputs
        ]

        teacher_concat = torch.cat(teacher_outputs, dim=0)
        if self.target_method == "center":
            teacher_prob_concat = F.softmax(
                (teacher_concat - self.center) / teacher_temp, dim=-1
            ).detach()
        else:
            teacher_prob_concat = self._sinkhorn_knopp(teacher_concat, teacher_temp).detach()
        view_sizes = [t.shape[0] for t in teacher_outputs]
        teacher_probs = list(teacher_prob_concat.split(view_sizes, dim=0))

        total_loss = 0.0
        n_terms = 0
        for teacher_prob in teacher_probs:
            for student_log_prob in student_log_probs:
                total_loss = total_loss - (teacher_prob * student_log_prob).sum(dim=-1).mean()
                n_terms += 1
        loss = total_loss / max(n_terms, 1)

        with torch.no_grad():
            student_concat = torch.cat(student_outputs, dim=0)
            student_prob_concat = torch.softmax(student_concat / self.student_temp, dim=-1)
            mean_teacher_prob = teacher_prob_concat.mean(dim=0)
            effective_prototypes = torch.exp(
                -(mean_teacher_prob * mean_teacher_prob.clamp_min(1e-8).log()).sum()
            )
            teacher_entropy = self._entropy(teacher_prob_concat)
            assignment_perplexity = torch.exp(teacher_entropy)
            metrics = {
                "teacher_entropy": teacher_entropy,
                "student_entropy": self._entropy(student_prob_concat),
                "teacher_max_prob": teacher_prob_concat.max(dim=-1).values.mean(),
                "student_max_prob": student_prob_concat.max(dim=-1).values.mean(),
                "center_norm": self.center.norm(),
                "prototype_usage": (mean_teacher_prob > (1.0 / teacher_prob_concat.shape[-1])).float().sum(),
                "effective_prototypes": effective_prototypes,
                "assignment_perplexity": assignment_perplexity,
                "teacher_temp": teacher_prob_concat.new_tensor(teacher_temp),
            }
            if update_center:
                if self.target_method == "center":
                    self.update_center(teacher_concat)
                self.step += 1
        return loss, metrics

    @staticmethod
    def _entropy(probabilities):
        return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1).mean()

    @torch.no_grad()
    def update_center(self, teacher_outputs):
        batch_center = teacher_outputs.mean(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)

    @torch.no_grad()
    def _sinkhorn_knopp(self, teacher_logits, teacher_temp):
        """
        Balanced soft assignment over prototypes via Sinkhorn-Knopp.
        """
        logits = teacher_logits.float()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        Q = torch.exp(logits / teacher_temp).t()  # [K, N_local]
        K, N_local = Q.shape
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        N_total = N_local * world_size

        sum_Q = Q.sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q = Q / sum_Q.clamp_min(1e-12)

        for _ in range(self.sinkhorn_iters):
            sum_rows = Q.sum(dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_rows)
            Q = Q / sum_rows.clamp_min(1e-12)
            Q = Q / K

            sum_cols = Q.sum(dim=0, keepdim=True)
            Q = Q / sum_cols.clamp_min(1e-12)
            Q = Q / N_total

        Q = Q * N_total
        return Q.t().to(teacher_logits.dtype)
