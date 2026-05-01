"""Reconstruction quality evaluator.

Computes rFID, Inception Score, PSNR, and SSIM metrics.
"""

import warnings
from typing import Optional, Mapping, Text

import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from .inception import get_inception_model


def get_covariance(sigma: torch.Tensor, total: torch.Tensor, num_examples: int) -> torch.Tensor:
    """Computes covariance from accumulated statistics."""
    if num_examples == 0:
        return torch.zeros_like(sigma)
    sub_matrix = torch.outer(total, total)
    sub_matrix = sub_matrix / num_examples
    return (sigma - sub_matrix) / (num_examples - 1)


class Evaluator:
    """Evaluates image reconstruction quality with multiple metrics.

    Supports:
    - rFID (reconstruction Frechet Inception Distance)
    - Inception Score
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - Codebook usage and entropy (for VQ models)

    Args:
        device: Compute device.
        enable_rfid: Whether to compute rFID.
        enable_inception_score: Whether to compute IS.
        enable_codebook_usage_measure: Whether to measure codebook utilization.
        enable_codebook_entropy_measure: Whether to measure codebook entropy.
        num_codebook_entries: Number of codebook entries (for VQ models).
    """

    def __init__(
        self,
        device,
        enable_rfid: bool = True,
        enable_inception_score: bool = True,
        enable_codebook_usage_measure: bool = False,
        enable_codebook_entropy_measure: bool = False,
        num_codebook_entries: int = 1024
    ):
        self._device = device
        self._enable_rfid = enable_rfid
        self._enable_inception_score = enable_inception_score
        self._enable_codebook_usage_measure = enable_codebook_usage_measure
        self._enable_codebook_entropy_measure = enable_codebook_entropy_measure
        self._num_codebook_entries = num_codebook_entries

        self._inception_model = None
        self._is_num_features = 0
        self._rfid_num_features = 0
        if self._enable_inception_score or self._enable_rfid:
            self._rfid_num_features = 2048
            self._is_num_features = 1008
            self._inception_model = get_inception_model().to(self._device)
            self._inception_model.eval()
        self._is_eps = 1e-16
        self._rfid_eps = 1e-6

        self.reset_metrics()

    def reset_metrics(self):
        """Resets all accumulated metrics."""
        self._num_examples = 0
        self._num_updates = 0

        self._is_prob_total = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._is_total_kl_d = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_real_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_real_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_fake_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_fake_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )

        self._set_of_codebook_indices = set()
        self._codebook_frequencies = torch.zeros(
            (self._num_codebook_entries), dtype=torch.float64, device=self._device
        )

        self.psnr_val_rgb = []
        self.ssim_val_rgb = []

    def update(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        codebook_indices: Optional[torch.Tensor] = None
    ):
        """Update metrics with a batch of real and reconstructed images.

        Args:
            real_images: Original images (ImageNet-normalized).
            fake_images: Reconstructed images (tanh range [-1, 1]).
            codebook_indices: Optional VQ codebook indices.
        """
        batch_size = real_images.shape[0]
        self._num_examples += batch_size
        self._num_updates += 1

        if self._enable_inception_score or self._enable_rfid:
            fake_inception_images = ((fake_images + 1) / 2 * 255).to(torch.uint8)
            features_fake = self._inception_model(fake_inception_images)
            inception_logits_fake = features_fake["logits_unbiased"]
            inception_probabilities_fake = F.softmax(inception_logits_fake, dim=-1)

        if self._enable_inception_score:
            probabiliies_sum = torch.sum(inception_probabilities_fake, 0, dtype=torch.float64)
            log_prob = torch.log(inception_probabilities_fake + self._is_eps)
            if log_prob.dtype != inception_probabilities_fake.dtype:
                log_prob = log_prob.to(inception_probabilities_fake)
            kl_sum = torch.sum(inception_probabilities_fake * log_prob, 0, dtype=torch.float64)
            self._is_prob_total += probabiliies_sum
            self._is_total_kl_d += kl_sum

        if self._enable_rfid:
            std = torch.tensor([0.229, 0.224, 0.225]).to(real_images.device)
            mean = torch.tensor([0.485, 0.456, 0.406]).to(real_images.device)
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
            real_images = real_images * std + mean
            real_inception_images = (real_images * 255).to(torch.uint8)
            features_real = self._inception_model(real_inception_images)

            if (features_real['2048'].shape[0] != features_fake['2048'].shape[0] or
                    features_real['2048'].shape[1] != features_fake['2048'].shape[1]):
                raise ValueError("Number of features should be equal for real and fake.")

            for f_real, f_fake in zip(features_real['2048'], features_fake['2048']):
                self._rfid_real_total += f_real
                self._rfid_fake_total += f_fake
                self._rfid_real_sigma += torch.outer(f_real, f_real)
                self._rfid_fake_sigma += torch.outer(f_fake, f_fake)

            rgb_restored = ((fake_images + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
            rgb_gt = real_images.permute(0, 2, 3, 1).cpu().numpy()
            for rgb_real, rgb_fake in zip(rgb_gt, rgb_restored):
                psnr = psnr_loss(rgb_fake, rgb_real)
                ssim = ssim_loss(rgb_fake, rgb_real, multichannel=True, data_range=2.0, channel_axis=-1)
                self.psnr_val_rgb.append(psnr)
                self.ssim_val_rgb.append(ssim)

        if self._enable_codebook_usage_measure and codebook_indices is not None:
            self._set_of_codebook_indices |= set(torch.unique(codebook_indices, sorted=False).tolist())

        if self._enable_codebook_entropy_measure and codebook_indices is not None:
            entries, counts = torch.unique(codebook_indices, sorted=False, return_counts=True)
            self._codebook_frequencies.index_add_(0, entries.int(), counts.double())

    def result(self) -> Mapping[Text, torch.Tensor]:
        """Compute and return all evaluation metrics."""
        eval_score = {}

        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")

        if self._enable_inception_score:
            mean_probs = self._is_prob_total / self._num_examples
            log_mean_probs = torch.log(mean_probs + self._is_eps)
            if log_mean_probs.dtype != self._is_prob_total.dtype:
                log_mean_probs = log_mean_probs.to(self._is_prob_total)
            excess_entropy = self._is_prob_total * log_mean_probs
            avg_kl_d = torch.sum(self._is_total_kl_d - excess_entropy) / self._num_examples
            inception_score = torch.exp(avg_kl_d).item()
            eval_score["InceptionScore"] = inception_score

        if self._enable_rfid:
            mu_real = self._rfid_real_total / self._num_examples
            mu_fake = self._rfid_fake_total / self._num_examples
            sigma_real = get_covariance(self._rfid_real_sigma, self._rfid_real_total, self._num_examples)
            sigma_fake = get_covariance(self._rfid_fake_sigma, self._rfid_fake_total, self._num_examples)

            mu_real, mu_fake = mu_real.cpu(), mu_fake.cpu()
            sigma_real, sigma_fake = sigma_real.cpu(), sigma_fake.cpu()

            diff = mu_real - mu_fake
            covmean, _ = linalg.sqrtm(sigma_real.mm(sigma_fake).numpy(), disp=False)
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError(f"Imaginary component {m}")
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            if not np.isfinite(covmean).all():
                tr_covmean = np.sum(np.sqrt((
                    (np.diag(sigma_real) * self._rfid_eps) * (np.diag(sigma_fake) * self._rfid_eps))
                    / (self._rfid_eps * self._rfid_eps)
                ))

            rfid = float(
                diff.dot(diff).item() + torch.trace(sigma_real)
                + torch.trace(sigma_fake) - 2 * tr_covmean
            )
            if torch.isnan(torch.tensor(rfid)) or torch.isinf(torch.tensor(rfid)):
                warnings.warn("The product of covariance of train and test features is out of bounds.")

            eval_score["rFID"] = rfid

            psnr_val_rgb = sum(self.psnr_val_rgb) / len(self.psnr_val_rgb)
            ssim_val_rgb = sum(self.ssim_val_rgb) / len(self.ssim_val_rgb)
            eval_score['psnr'] = psnr_val_rgb
            eval_score['ssim'] = ssim_val_rgb

        if self._enable_codebook_usage_measure:
            usage = float(len(self._set_of_codebook_indices)) / self._num_codebook_entries
            eval_score["CodebookUsage"] = usage

        if self._enable_codebook_entropy_measure:
            probs = self._codebook_frequencies / self._codebook_frequencies.sum()
            entropy = (-torch.log2(probs + 1e-8) * probs).sum()
            eval_score["CodebookEntropy"] = entropy

        return eval_score
