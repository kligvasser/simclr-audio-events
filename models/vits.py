import torch.nn as nn
import torchvision.models as torch_models
import torchaudio.transforms as T
import numpy as np


class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class ViTSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, n_fft, hop_length, normalized):
        super(ViTSimCLR, self).__init__()
        self.vit_dict = {
            "vit_b_16": torch_models.vit_b_16(weights=None, num_classes=out_dim),
            "vit_b_32": torch_models.vit_b_32(weights=None, num_classes=out_dim),
        }

        self.spectogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, normalized=normalized)
        self.time_masking = T.TimeMasking(time_mask_param=50)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=50)

        self.backbone = self._get_basemodel(base_model)
        hidden_dim = self.backbone.hidden_dim

        # add projector head
        self.project_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.backbone.heads = nn.Identity()

        # add spec head
        self.spec_head = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 3, 7, 1, 3))

    def _get_basemodel(self, model_name):
        try:
            model = self.vit_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: {}".format(
                    ",".join(list(self.vit_dict.keys()))
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

    base_model = "vit_b_32"
    out_dim = 512
    n_fft = 446
    hop_length = int(np.ceil(10 * 8000 / 224))
    normalized = False

    m = ViTSimCLR(base_model, out_dim, n_fft, hop_length, normalized)
    x = torch.randn(4, 1, 10 * 8000)
    e = m(x)
    print(e.size())
