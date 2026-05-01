"""Learning rate schedulers with warm-up support.

Reference:
    https://raw.githubusercontent.com/huggingface/open-muse/vqgan-finetuning/muse/lr_schedulers.py
"""

import math
from enum import Enum
from typing import Optional, Union

import torch


class SchedulerType(Enum):
    COSINE = "cosine"
    CONSTANT = "constant"


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Creates a cosine learning rate schedule with warm-up and ending learning rate.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of periods of the cosine function in a schedule.
        last_epoch: The index of the last epoch when resuming training.
        base_lr: The base learning rate.
        end_lr: The final learning rate.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * ratio) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Creates a constant learning rate schedule with warm-up.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps (unused, kept for API consistency).
        base_lr: The base learning rate (unused, kept for API consistency).
        end_lr: The final learning rate (unused, kept for API consistency).

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule_with_warmup,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
):
    """Retrieves a learning rate scheduler by name.

    Args:
        name: The name of the scheduler to retrieve ("cosine" or "constant").
        optimizer: The optimizer to use with the scheduler.
        num_warmup_steps: The number of warmup steps.
        num_training_steps: The total number of training steps.
        base_lr: The base learning rate.
        end_lr: The final learning rate.

    Returns:
        An instance of torch.optim.lr_scheduler.LambdaLR.

    Raises:
        ValueError: If num_warmup_steps or num_training_steps is not provided.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        base_lr=base_lr,
        end_lr=end_lr,
    )
