import torch
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, num_views=2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.num_views = num_views

    def info_nce_loss(self, features):
        batch_size = len(features) // self.num_views
        device = features.device

        labels = torch.cat([torch.arange(batch_size) for _ in range(self.num_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature

        return logits, labels

    def forward(self, features):
        logits, labels = self.info_nce_loss(features)
        return logits, labels
