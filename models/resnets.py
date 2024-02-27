import torch.nn as nn
import torchvision.models as torch_models
import torchaudio.transforms as T
import numpy as np


class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, n_fft, hop_length, normalized):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": torch_models.resnet18(weights=None, num_classes=out_dim),
            "resnet50": torch_models.resnet50(weights=None, num_classes=out_dim),
            "resnet101": torch_models.resnet101(weights=None, num_classes=out_dim),
            "resnext50_32x4d": torch_models.resnext50_32x4d(weights=None, num_classes=out_dim),
            "resnext101_32x8d": torch_models.resnext101_32x8d(weights=None, num_classes=out_dim),
        }

        self.spectogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, normalized=normalized)
        self.time_masking = T.TimeMasking(time_mask_param=50)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=50)

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add projector head
        self.project_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp, bias=False),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim, bias=False),
        )
        self.backbone.fc = nn.Identity()

        # add spec head
        self.spec_head = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 3, 7, 1, 3))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: {}".format(
                    ",".join(list(self.resnet_dict.keys()))
                )
            )
        else:
            return model

    def forward(self, x):
        e = self.embedding(x)
        return self.project_head(e)

    def embedding(self, x):
        x = self.spectogram(x)
        if self.training:
            x = self.time_masking(x)
            x = self.freq_masking(x)

        x = self.spec_head(x)
        return self.backbone(x)


if __name__ == "__main__":
    import torch

    base_model = "resnet50"
    out_dim = 512
    n_fft = 510
    hop_length = int(np.ceil(10 * 8000 / 256))
    normalized = False

    m = ResNetSimCLR(base_model, out_dim, n_fft, hop_length, normalized)
    x = torch.randn(4, 1, 10 * 8000)
    e = m(x)
    print(e.size())
