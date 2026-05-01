"""Exponential Moving Average (EMA) model wrapper.

Reference:
    https://github.com/huggingface/open-muse/blob/64e1afe033717d795866ab8204484705cd4dc3f7/muse/modeling_ema.py
"""

import copy
from typing import Any, Iterable, Optional, Union

import torch


class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        update_every: int = 1,
        current_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        **model_config_kwargs
    ):
        """
        Args:
            parameters: The parameters to track.
            decay: The decay factor for the exponential moving average.
            min_decay: The minimum decay factor.
            update_after_step: Steps to wait before starting EMA updates.
            update_every: Steps between each EMA update.
            current_step: The current training step.
            use_ema_warmup: Whether to use EMA warmup.
            inv_gamma: Inverse multiplicative factor of EMA warmup (default: 1).
            power: Exponential factor of EMA warmup (default: 2/3).

        Notes on EMA Warmup:
            gamma=1, power=2/3 are good for models trained 1M+ steps.
            gamma=1, power=3/4 for models trained fewer steps.
        """
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = current_step
        self.cur_decay_value = None

        self.model_cls = model_cls
        self.model_config_kwargs = model_config_kwargs

    @classmethod
    def from_pretrained(cls, checkpoint, model_cls, **model_config_kwargs) -> "EMAModel":
        model = model_cls(**model_config_kwargs)
        model.load_pretrained_weight(checkpoint)
        ema_model = cls(model.parameters(), model_cls=model_cls, **model_config_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")
        if self.model_config_kwargs is None:
            raise ValueError("`save_pretrained` can only be used if `model_config_kwargs` was defined at __init__.")

        model = self.model_cls(**self.model_config_kwargs)
        self.copy_to(model.parameters())
        model.save_pretrained_weight(path)

    def set_step(self, optimization_step: int):
        self.optimization_step = optimization_step

    def get_decay(self, optimization_step: int) -> float:
        """Computes the decay factor for the exponential moving average."""
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        parameters = list(parameters)
        self.optimization_step += 1

        if (self.optimization_step - 1) % self.update_every != 0:
            return

        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Copies current averaged parameters into the given collection."""
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        """Moves internal buffers to `device`."""
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        """Returns the state of the ExponentialMovingAverage as a dict."""
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Save the current parameters for restoring later."""
        self.temp_stored_params = [param.detach().cpu().clone() for param in parameters]

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Restore the parameters stored with the `store` method."""
        if self.temp_stored_params is None:
            raise RuntimeError("This EMAModel has no `store()`ed weights to `restore()`")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the EMA state."""
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")
