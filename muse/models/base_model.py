"""Base model class with pretrained weight save/load utilities.

Reference:
    https://github.com/huggingface/open-muse/blob/main/muse/modeling_utils.py
"""

import os
from typing import Union, Callable, Dict, Optional

import torch


class BaseModel(torch.nn.Module):
    """Base class for all MUSE models with weight I/O support."""

    def __init__(self):
        super().__init__()

    def save_pretrained_weight(
        self,
        save_directory: Union[str, os.PathLike],
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Saves model weights to a directory.

        Args:
            save_directory: Directory to save weights. Created if nonexistent.
            save_function: Custom save function (default: torch.save).
            state_dict: State dict to save (default: self.state_dict()).
        """
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        if state_dict is None:
            state_dict = model_to_save.state_dict()
        weights_name = "pytorch_model.bin"

        save_function(state_dict, os.path.join(save_directory, weights_name))
        print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def load_pretrained_weight(
        self,
        pretrained_model_path: Union[str, os.PathLike],
        strict_loading: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """Loads pretrained weights from a file or directory.

        Args:
            pretrained_model_path: Path to a weight file or directory
                containing 'pytorch_model.bin'.
            strict_loading: Whether to strictly enforce state_dict keys match.
            torch_dtype: Optional dtype to cast the model to after loading.

        Raises:
            ValueError: If pretrained_model_path does not exist.
        """
        if os.path.isfile(pretrained_model_path):
            model_file = pretrained_model_path
        elif os.path.isdir(pretrained_model_path):
            pretrained_model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.isfile(pretrained_model_path):
                model_file = pretrained_model_path
            else:
                raise ValueError(f"{pretrained_model_path} does not exist")
        else:
            raise ValueError(f"{pretrained_model_path} does not exist")

        checkpoint = torch.load(model_file, map_location="cpu")
        msg = self.load_state_dict(checkpoint, strict=strict_loading)
        print(f"Loading weight from {model_file}, msg: {msg}")

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, "
                f"e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            self.to(torch_dtype)

        self.eval()

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """Gets the number of parameters in the module.

        Args:
            only_trainable: Whether to only count trainable parameters.
            exclude_embeddings: Whether to exclude embedding parameters.

        Returns:
            Number of (trainable) parameters.
        """
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters()
                if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)
