from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class HashingHead(nn.Module):
    """
    Learns compact retrieval codes from continuous embeddings.

    During training it returns a tanh-relaxed code in [-1, 1]. During inference,
    callers can use hard_codes() to get sign-binarized {-1, +1} codes suitable
    for packing into FAISS binary indexes.
    """

    def __init__(
        self,
        in_dim: int = 384,
        num_bits: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_bits % 8 != 0:
            raise ValueError("num_bits must be divisible by 8 for FAISS binary indexes.")

        self.in_dim = in_dim
        self.num_bits = num_bits
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bits),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.net(embeddings)
        return torch.tanh(logits)

    @torch.no_grad()
    def hard_codes(self, embeddings: torch.Tensor) -> torch.Tensor:
        soft_codes = self.forward(embeddings)
        return torch.where(
            soft_codes >= 0,
            torch.ones_like(soft_codes),
            -torch.ones_like(soft_codes),
        )

    def quantization_loss(self, soft_codes: torch.Tensor) -> torch.Tensor:
        return torch.mean((soft_codes.abs() - 1.0) ** 2)


def similarity_preserving_hash_loss(
    soft_codes: torch.Tensor,
    labels: torch.Tensor,
    quantization_weight: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple supervised hashing loss for multi-label or single-label batches.

    It aligns code cosine similarity with semantic relevance and adds a
    quantization penalty that pushes tanh outputs toward binary values.
    """
    codes = F.normalize(soft_codes, dim=1)
    code_similarity = codes @ codes.T

    if labels.ndim == 2:
        relevance = (labels.float() @ labels.float().T > 0).float()
    else:
        relevance = (labels.view(-1, 1) == labels.view(1, -1)).float()

    target = relevance * 2.0 - 1.0
    similarity_loss = F.mse_loss(code_similarity, target)
    quantization = torch.mean((soft_codes.abs() - 1.0) ** 2)
    total = similarity_loss + quantization_weight * quantization
    return total, similarity_loss, quantization

