"""InceptionV3 feature extractor for FID and Inception Score computation.

Reference:
    https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py
"""

import torch
import torch.nn.functional as F

from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.helpers import vassert
from torch_fidelity.feature_extractor_inceptionv3 import (
    BasicConv2d, InceptionA, InceptionB, InceptionC,
    InceptionD, InceptionE_1, InceptionE_2
)
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class FeatureExtractorInceptionV3(FeatureExtractorBase):
    """InceptionV3 feature extractor for 2D RGB 24bit images.

    Provides features at layers: '64', '192', '768', '2048', 'logits_unbiased', 'logits'.
    """

    INPUT_IMAGE_SIZE = 299

    def __init__(self, name, features_list, **kwargs):
        super().__init__(name, features_list)
        self.feature_extractor_internal_dtype = torch.float64

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE_1(1280)
        self.Mixed_7c = InceptionE_2(2048)
        self.AvgPool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = torch.nn.Linear(2048, 1008)

        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
        self.load_state_dict(state_dict)

        self.to(self.feature_extractor_internal_dtype)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        vassert(torch.is_tensor(x) and x.dtype == torch.uint8,
                'Expecting image as torch.Tensor with dtype=torch.uint8')
        vassert(x.dim() == 4 and x.shape[1] == 3, f'Input is not Bx3xHxW: {x.shape}')

        features = {}
        remaining_features = self.features_list.copy()

        x = x.to(self.feature_extractor_internal_dtype)

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x, size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE), align_corners=False,
        )

        x = (x - 128) / 128

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)

        if '64' in remaining_features:
            features['64'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('64')
            if len(remaining_features) == 0:
                return features

        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)

        if '192' in remaining_features:
            features['192'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1)
            remaining_features.remove('192')
            if len(remaining_features) == 0:
                return features

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        if '768' in remaining_features:
            features['768'] = F.adaptive_avg_pool2d(x, output_size=(1, 1)).squeeze(-1).squeeze(-1).to(torch.float32)
            remaining_features.remove('768')
            if len(remaining_features) == 0:
                return features

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)

        if '2048' in remaining_features:
            features['2048'] = x
            remaining_features.remove('2048')
            if len(remaining_features) == 0:
                return features

        if 'logits_unbiased' in remaining_features:
            x = x.mm(self.fc.weight.T)
            features['logits_unbiased'] = x
            remaining_features.remove('logits_unbiased')
            if len(remaining_features) == 0:
                return features
            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)

        features['logits'] = x
        return features

    @staticmethod
    def get_provided_features_list():
        return '64', '192', '768', '2048', 'logits_unbiased', 'logits'

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return {
            'isc': 'logits_unbiased',
            'fid': '2048',
            'kid': '2048',
            'prc': '2048',
        }[metric]

    @staticmethod
    def can_be_compiled():
        return True

    @staticmethod
    def get_dummy_input_for_compile():
        return (torch.rand([1, 3, 4, 4]) * 255).to(torch.uint8)


def get_inception_model():
    """Create InceptionV3 model for FID and IS computation."""
    model = FeatureExtractorInceptionV3("inception_model", ["2048", "logits_unbiased"])
    return model
