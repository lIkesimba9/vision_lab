"""MAE — Masked Autoencoder (He 2021, https://arxiv.org/abs/2111.06377).

Флагманский masked image modeling: 75% патчей выбрасывается, энкодер видит
ТОЛЬКО видимые патчи (в этом экономия compute — главное отличие от SimMIM),
лёгкий ViT-декодер восстанавливает пиксели замаскированных патчей, MSE только
на маске (с пер-патч нормализацией таргета, ``norm_pix_loss``).

Требует timm ViT (нужны ``patch_embed``/``blocks``/``norm``/``pos_embed`` —
энкодер прогоняется по видимому подмножеству токенов вручную). Для Swin/CNN
используйте :class:`~vision_lab.ssl.simmim.SimMIM`. Аугментации минимальны
(рецепт ``mim_v1``) — маскирование заменяет тяжёлые вьюхи.

Без учителя (``momentum_update`` — no-op). ``extract_embeddings`` — pooled
энкодер-токены без маскирования (декодер после претрейна выбрасывается).
"""

from __future__ import annotations

import torch
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block
from torch import nn

from vision_lab.models.backbones import TokenBackbone
from vision_lab.ssl.base import SSLMethod
from vision_lab.ssl.components import patchify


def random_masking(tokens: torch.Tensor, mask_ratio: float
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Пер-сэмпл случайное маскирование токенов (перестановкой, как в статье).

    Возвращает ``(visible, mask, ids_restore)``: видимые токены
    ``(B, len_keep, D)``, маску ``(B, N)`` (True = замаскирован) и перестановку
    для восстановления порядка. Гарантия «не всё и не ничего»: остаётся
    >= 1 видимый и >= 1 замаскированный токен.
    """
    b, n, d = tokens.shape
    len_keep = min(n - 1, max(1, int(round(n * (1.0 - mask_ratio)))))
    noise = torch.rand(b, n, device=tokens.device)
    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    visible = tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.ones(b, n, dtype=torch.bool, device=tokens.device)
    mask[:, :len_keep] = False
    mask = mask.gather(1, ids_restore)
    return visible, mask, ids_restore


class MAE(SSLMethod):
    """MAE: ViT-энкодер по видимым патчам + лёгкий ViT-декодер по всем.

    Компоненты передаются инстанцированными:
        ``backbone`` — :class:`TokenBackbone` над timm ViT;
        ``views``    — :class:`~vision_lab.ssl.gpu_augs.MultiViewAugment`
                       (1 вьюха, рецепт ``mim_v1``).

    ``image_size`` фиксирует сетку патчей (размер decoder_pos_embed) — обязан
    совпадать с ``image_size`` вьюх. Дефолты декодера уменьшены относительно
    статьи (512×8) — под средние датасеты; для ImageNet-масштаба поднимите.
    """

    def __init__(self, backbone: TokenBackbone, views: nn.Module, image_size: int,
                 mask_ratio: float = 0.75, norm_pix_loss: bool = True,
                 decoder_dim: int = 256, decoder_depth: int = 4,
                 decoder_heads: int = 8):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio={mask_ratio} должен быть в (0, 1)")
        net = backbone.net
        for attr in ("patch_embed", "blocks", "norm", "pos_embed"):
            if getattr(net, attr, None) is None:
                raise ValueError(
                    f"MAE требует timm ViT (нет атрибута {attr!r}); "
                    "для Swin/CNN используйте SimMIM"
                )
        if backbone.num_prefix_tokens > 1:
            raise ValueError("MAE поддерживает ViT с <= 1 префикс-токеном (cls); "
                             "register-токены не поддержаны")

        self.backbone = backbone
        self.views = views
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        gh, gw = backbone.grid_size((image_size, image_size))
        self.grid = (gh, gw)
        self.n_patches = gh * gw
        if self.n_patches < 2:
            raise ValueError(
                f"image_size={image_size} даёт сетку {self.grid} — меньше 2 патчей, "
                "маскировать нечего (увеличьте image_size или уменьшите патч)"
            )
        self.patch_px = image_size // gh

        prefix = backbone.num_prefix_tokens
        self.decoder_embed = nn.Linear(backbone.out_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, prefix + self.n_patches, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_px * self.patch_px * 3)

    @torch.no_grad()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Pooled энкодер-токены (cls/mean) БЕЗ маскирования; декодер не участвует."""
        return self.backbone.embed(images)

    # -- энкодер по видимым токенам (ручной проход по внутренностям timm ViT) --
    def _encode_visible(self, x: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        net = self.backbone.net
        prefix = self.backbone.num_prefix_tokens
        # у no_embed_class-вариантов ViT pos_embed не содержит строк префикс-токенов
        embed_prefix = 0 if getattr(net, "no_embed_class", False) else prefix

        tok = net.patch_embed(x)
        if tok.dim() == 4:  # dynamic_img_size ViT: (B, H, W, D) -> (B, N, D)
            tok = tok.reshape(tok.shape[0], -1, tok.shape[-1])
        pos = resample_abs_pos_embed(net.pos_embed, new_size=list(self.grid),
                                     num_prefix_tokens=embed_prefix)
        tok = tok + pos[:, embed_prefix:, :]

        visible, mask, ids_restore = random_masking(tok, self.mask_ratio)
        if prefix:
            cls = net.cls_token
            if embed_prefix:
                cls = cls + pos[:, :1, :]
            visible = torch.cat([cls.expand(visible.shape[0], -1, -1), visible], dim=1)
        for blk in net.blocks:
            visible = blk(visible)
        return net.norm(visible), mask, ids_restore

    def _decode(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        prefix = self.backbone.num_prefix_tokens
        y = self.decoder_embed(latent)
        b = y.shape[0]
        n_masked = self.n_patches - (y.shape[1] - prefix)
        mask_tokens = self.mask_token.expand(b, n_masked, -1)
        patches = torch.cat([y[:, prefix:, :], mask_tokens], dim=1)
        patches = patches.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        y = torch.cat([y[:, :prefix, :], patches], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            y = blk(y)
        return self.decoder_pred(self.decoder_norm(y)[:, prefix:, :])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        x = self.views(batch["image"]).globals[0]
        grid = self.backbone.grid_size((x.shape[-2], x.shape[-1]))
        if grid != self.grid:
            raise ValueError(
                f"Сетка патчей вьюхи {grid} != сетке из image_size {self.grid} — "
                "image_size MAE обязан совпадать с image_size вьюх"
            )

        latent, mask, ids_restore = self._encode_visible(x)
        pred = self._decode(latent, ids_restore)          # (B, N, ph*ph*3)

        target = patchify(x, self.grid)
        if self.norm_pix_loss:  # пер-патч нормализация таргета (стабилизирует, §статьи)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
        m = mask.float()
        loss = (loss_per_patch * m).sum() / m.sum().clamp(min=1.0)
        return {"mae_loss": loss, "total_loss": loss}
