"""WebDataset-based data loading for MUSE training.

Supports two modes:
- ImageNet mode: expects .cls files for class labels
- DataComp/Text mode: expects .txt or .json files for text captions
"""

import math
import json
import glob
from typing import List, Union, Text, Dict, Any

import webdataset as wds
import torch
from torch.utils.data import default_collate
from torchvision import transforms
from PIL import Image

# Prevent DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None


# ==============================================================================
# 1. Helper Functions
# ==============================================================================

def identity(x):
    return x


def filter_keys(key_set):
    """Keep only specified keys to reduce memory usage."""
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}
    return _f


def robust_text_extractor(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Smart text extraction: supports .txt files and .json caption/text fields."""
    text = ""
    # 1. Try direct text files
    for key in ['txt', 'text', 'caption']:
        if key in sample:
            val = sample[key]
            if isinstance(val, bytes):
                text = val.decode("utf-8")
            else:
                text = str(val)
            break

    # 2. Fallback to JSON parsing
    if not text and 'json' in sample:
        try:
            json_data = sample['json']
            if isinstance(json_data, bytes):
                json_data = json.loads(json_data.decode("utf-8"))
            if isinstance(json_data, dict):
                text = json_data.get('caption', "") or json_data.get('text', "")
        except Exception:
            pass

    sample['text'] = text
    return sample


def filter_by_res_ratio(min_res=256, min_ratio=0.5, max_ratio=2.0):
    """Optional: filter images by resolution and aspect ratio."""
    def _f(sample):
        if 'json' not in sample:
            return True
        try:
            cfg = sample['json']
            if isinstance(cfg, bytes):
                cfg = json.loads(cfg.decode("utf-8"))
            h = cfg.get('original_height', 256)
            w = cfg.get('original_width', 256)
            if h is None or w is None:
                return True
            ratio = h / (w + 1e-6)
            longer_side = max(h, w)
            return (ratio >= min_ratio) and (ratio <= max_ratio) and (longer_side >= min_res)
        except Exception:
            return True
    return _f


# ==============================================================================
# 2. Image Transforms
# ==============================================================================

class ImageTransform:
    """Train and eval image transforms with ImageNet normalization."""

    def __init__(
        self,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
    ):
        interpolation = transforms.InterpolationMode.BICUBIC

        # Train Transform
        train_ops = [
            transforms.Resize(resize_shorter_edge, interpolation=interpolation, antialias=True)
        ]
        if random_crop:
            train_ops.append(transforms.RandomCrop(crop_size))
        else:
            train_ops.append(transforms.CenterCrop(crop_size))

        if random_flip:
            train_ops.append(transforms.RandomHorizontalFlip())

        train_ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        self.train_transform = transforms.Compose(train_ops)

        # Eval Transform
        self.eval_transform = transforms.Compose([
            transforms.Resize(crop_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])


# ==============================================================================
# 3. Main Dataset Class
# ==============================================================================

class TextImageDataset:
    """WebDataset-based dataset for MUSE training.

    Supports multi-dataset mixing with configurable sample ratios.

    Args:
        train_shards_path: Path(s) to training tar shards.
        eval_shards_path: Path(s) to evaluation tar shards.
        num_train_examples: Total training examples per epoch.
        per_gpu_batch_size: Batch size per GPU.
        global_batch_size: Total batch size across all GPUs.
        num_workers_per_gpu: DataLoader workers per GPU.
        dataset_with_class_label: True for ImageNet mode (expects .cls).
        dataset_with_text_label: True for DataComp mode (expects .txt/.json).
        sample_ratio: Mixing ratios for multiple dataset paths.
    """

    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int = 4,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
        dataset_with_class_label: bool = True,
        dataset_with_text_label: bool = False,
        res_ratio_filtering: bool = False,
        sample_ratio: List[float] = [],
    ):
        transform = ImageTransform(
            resize_shorter_edge, crop_size, random_crop, random_flip,
            normalize_mean, normalize_std
        )

        # --- Pipeline Construction ---
        if dataset_with_class_label:
            train_processing_pipeline = [
                wds.decode("pil", handler=wds.warn_and_continue),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    class_id="cls",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["image", "class_id", "__key__"]))),
                wds.map_dict(
                    image=transform.train_transform,
                    class_id=lambda x: int(x),
                    handler=wds.warn_and_continue,
                ),
            ]
        elif dataset_with_text_label:
            train_processing_pipeline = [
                wds.select(filter_by_res_ratio()) if res_ratio_filtering else wds.map(identity),
                wds.decode("pil", handler=wds.warn_and_continue),
                wds.map(robust_text_extractor),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys(set(["image", "text", "__key__"]))),
                wds.map_dict(
                    image=transform.train_transform,
                    handler=wds.warn_and_continue,
                ),
            ]
        else:
            raise NotImplementedError("Must select either class_label or text_label mode")

        # Eval Pipeline
        test_processing_pipeline = [
            wds.decode("pil", handler=wds.warn_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(["image", "__key__"]))),
            wds.map_dict(
                image=transform.eval_transform,
                handler=wds.warn_and_continue,
            ),
        ]

        # --- Train Loader ---
        if isinstance(train_shards_path, str):
            train_shards_path = [train_shards_path]

        pipelines = []
        for urls in train_shards_path:
            if "{" not in urls and not urls.startswith("http"):
                tar_urls = glob.glob(urls)
            else:
                tar_urls = urls

            pipeline = [
                wds.ResampledShards(tar_urls),
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.shuffle(bufsize=5000, initial=1000),
                *train_processing_pipeline,
                wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
            ]
            pipelines.append(pipeline)

        calc_workers = max(num_workers_per_gpu, 1)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * calc_workers))
        num_batches = num_worker_batches * calc_workers
        num_samples = num_batches * global_batch_size

        if len(pipelines) > 1:
            if not sample_ratio:
                sample_ratio = [1.0 / len(pipelines)] * len(pipelines)
            self._train_dataset = wds.DataPipeline(
                wds.RandomMix(
                    [wds.DataPipeline(*p) for p in pipelines],
                    probs=sample_ratio
                )
            )
        else:
            self._train_dataset = wds.DataPipeline(*pipelines[0])

        self._train_dataset = self._train_dataset.with_epoch(num_worker_batches)

        use_persistent_workers = (num_workers_per_gpu > 0)

        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
        )
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # --- Eval Loader ---
        if eval_shards_path:
            eval_pipeline = [
                wds.SimpleShardList(eval_shards_path),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.ignore_and_continue),
                *test_processing_pipeline,
                wds.batched(per_gpu_batch_size, partial=True, collation_fn=default_collate),
            ]
            self._eval_dataset = wds.DataPipeline(*eval_pipeline)
            self._eval_dataloader = wds.WebLoader(
                self._eval_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers_per_gpu,
                pin_memory=True,
                persistent_workers=use_persistent_workers,
            )
        else:
            self._eval_dataset = None
            self._eval_dataloader = None

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader
