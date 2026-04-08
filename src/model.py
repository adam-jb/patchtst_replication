"""PatchTST model: patch embedding, BatchNorm encoder, supervised & pretrain heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Unfold univariate series into patches, project to d_model, add positional encoding.

    From the paper (page 4): "We pad S repeated numbers of the last value to the end
    of the original sequence before patching." This gives N = floor((L-P)/S) + 2.
    With L=336, P=16, S=8: N = floor(320/8) + 2 = 42 patches (PatchTST/42).
    """

    def __init__(self, seq_len: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Paper formula: pad S values at the end, giving N = floor((L-P)/S) + 2
        self.pad_len = stride  # always pad by stride (paper says "pad S repeated numbers")
        self.n_patches = (seq_len - patch_len) // stride + 2
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, seq_len)
        Returns:
            embedded: (batch, n_patches, d_model)
            raw_patches: (batch, n_patches, patch_len) — for self-supervised target
        """
        # Pad S values at the end (replicate last value, as described in paper)
        x = F.pad(x, (0, self.pad_len), mode='replicate')
        # Unfold into patches: (batch, 1, seq_len+S) -> (batch, 1, n_patches, patch_len)
        raw_patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        raw_patches = raw_patches.squeeze(1)  # (batch, n_patches, patch_len)
        embedded = self.proj(raw_patches) + self.pos_enc  # (batch, n_patches, d_model)
        return embedded, raw_patches


class BatchNormEncoderLayer(nn.Module):
    """TransformerEncoderLayer with BatchNorm1d instead of LayerNorm (post-norm)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _bn(self, norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm1d: (B, N, D) -> permute -> BN -> permute back."""
        return norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual + batchnorm (post-norm)
        attn_out, _ = self.self_attn(x, x, x)
        x = self._bn(self.norm1, x + self.dropout1(attn_out))
        # FFN + residual + batchnorm
        ffn_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self._bn(self.norm2, x + self.dropout2(ffn_out))
        return x


class PatchTSTEncoder(nn.Module):
    """Stack of BatchNorm encoder layers."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            BatchNormEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalizes each input sample by its own mean/std before the model,
    then denormalizes predictions to restore the original scale.
    Handles non-stationarity in time series.
    """

    def __init__(self, n_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, n_channels, 1))
            self.beta = nn.Parameter(torch.zeros(1, n_channels, 1))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, length)
            mode: 'norm' to normalize, 'denorm' to reverse
        """
        if mode == "norm":
            self._mean = x.mean(dim=-1, keepdim=True).detach()
            self._std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self._std + self._mean
        return x


class PatchTST(nn.Module):
    """Full PatchTST model for supervised forecasting."""

    def __init__(self, seq_len: int = 336, pred_len: int = 96, patch_len: int = 16,
                 stride: int = 8, d_model: int = 16, nhead: int = 4,
                 dim_feedforward: int = 128, dropout: float = 0.2,
                 n_layers: int = 3, n_channels: int = 7, use_revin: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(n_channels)
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model)
        self.encoder = PatchTSTEncoder(d_model, nhead, dim_feedforward, dropout, n_layers)
        n_patches = self.patch_embed.n_patches
        self.head = nn.Linear(n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, seq_len)
        Returns:
            pred: (batch, n_channels, pred_len)
        """
        B, M, L = x.shape
        # RevIN: normalize each sample by its own stats
        if self.use_revin:
            x = self.revin(x, mode="norm")
        # Channel independence: flatten channels into batch
        x = x.reshape(B * M, 1, L)              # (B*M, 1, L)
        embedded, _ = self.patch_embed(x)         # (B*M, N, D)
        encoded = self.encoder(embedded)          # (B*M, N, D)
        flat = encoded.flatten(1)                 # (B*M, N*D)
        pred = self.head(flat)                    # (B*M, pred_len)
        pred = pred.reshape(B, M, -1)            # (B, M, pred_len)
        # RevIN: denormalize predictions
        if self.use_revin:
            pred = self.revin(pred, mode="denorm")
        return pred


class PatchTSTForPretrain(nn.Module):
    """PatchTST with masked patch prediction head for self-supervised pretraining."""

    def __init__(self, seq_len: int = 336, patch_len: int = 16, stride: int = 16,
                 d_model: int = 16, nhead: int = 4, dim_feedforward: int = 128,
                 dropout: float = 0.2, n_layers: int = 3, mask_ratio: float = 0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbedding(seq_len, patch_len, stride, d_model)
        self.encoder = PatchTSTEncoder(d_model, nhead, dim_feedforward, dropout, n_layers)
        self.recon_head = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, seq_len) — single univariate channel
        Returns:
            pred_patches: (batch, n_masked, patch_len) — predicted raw values at masked positions
            target_patches: (batch, n_masked, patch_len) — actual raw values at masked positions
            mask: (batch, n_patches) — bool mask (True = masked)
        """
        embedded, raw_patches = self.patch_embed(x)  # (B, N, D), (B, N, P)
        B, N, D = embedded.shape
        n_masked = int(N * self.mask_ratio)

        # Generate random mask per sample
        noise = torch.rand(B, N, device=x.device)
        mask_indices = noise.argsort(dim=1)[:, :n_masked]  # (B, n_masked)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, mask_indices, True)

        # Zero out masked patches
        masked_embedded = embedded.clone()
        masked_embedded[mask] = 0.0

        # Encode and reconstruct
        encoded = self.encoder(masked_embedded)  # (B, N, D)
        recon = self.recon_head(encoded)          # (B, N, P)

        # Extract only masked positions
        pred_patches = recon[mask].reshape(B, n_masked, -1)
        target_patches = raw_patches[mask].reshape(B, n_masked, -1)

        return pred_patches, target_patches, mask


if __name__ == "__main__":
    # Shape verification
    print("=== Supervised model ===")
    model = PatchTST(seq_len=336, pred_len=96)
    x = torch.randn(2, 7, 336)
    y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
    print(f"Patches: {model.patch_embed.n_patches}, Params: {sum(p.numel() for p in model.parameters()):,}")

    # Channel independence test
    model.eval()
    x1 = torch.randn(1, 7, 336)
    x2 = x1.clone()
    x2[:, 3, :] += 1.0
    y1 = model(x1)
    y2 = model(x2)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6), "Channel independence violated!"
    assert not torch.allclose(y1[:, 3], y2[:, 3], atol=1e-6), "Channel 3 should differ!"
    print("Channel independence: PASSED")

    print("\n=== Pretrain model ===")
    pretrain_model = PatchTSTForPretrain(seq_len=336, stride=16)
    x_uni = torch.randn(4, 1, 336)
    pred_p, target_p, mask = pretrain_model(x_uni)
    print(f"Input: {tuple(x_uni.shape)}")
    print(f"Patches: {pretrain_model.patch_embed.n_patches}, Masked: {pred_p.shape[1]}")
    print(f"Pred patches: {tuple(pred_p.shape)}, Target patches: {tuple(target_p.shape)}")
    print(f"Params: {sum(p.numel() for p in pretrain_model.parameters()):,}")
