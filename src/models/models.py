import torch
import timm
from torch import nn
from torchvision import models


class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-Tiny encoder (обёртка над torchvision), classifier → Identity."""

    def __init__(
        self,
        weights="ConvNeXt_Tiny_Weights.DEFAULT",
    ):
        super().__init__()
        self.model = models.convnext_tiny(weights=weights)
        #self.feature_size = self.backbone.classifier[-1].in_features
        self.model.classifier = torch.nn.Identity()

        
        

    def forward(self, x):
        x = self.model(x) # bs, 768, 1, 1
        return x.squeeze()




class EfficientNetV2(nn.Module):
    """EfficientNetV2 (s/m/l) backbone. classifier → Identity (эмбеддинги) либо Linear(num_classes)."""

    _VARIANTS = {
        "s": (lambda w: models.efficientnet_v2_s(weights=w), "EfficientNet_V2_S_Weights"),
        "m": (lambda w: models.efficientnet_v2_m(weights=w), "EfficientNet_V2_M_Weights"),
        "l": (lambda w: models.efficientnet_v2_l(weights=w), "EfficientNet_V2_L_Weights"),
    }

    def __init__(
        self,
        weights="DEFAULT",
        num_classes=None,
        path_to_local_weigth=None,
        variant="s",
    ):
        super().__init__()
        ctor, weights_enum_name = self._VARIANTS[variant]
        weights_enum = getattr(models, weights_enum_name)
        use_weights = weights_enum.DEFAULT if weights else None
        self.model = ctor(use_weights)
        # feature_size одинаков (1280) для s/m/l: self.model.classifier[-1].in_features

        if path_to_local_weigth is not None:
            ckpt = torch.load(
                path_to_local_weigth,
                map_location="cpu",
                weights_only=True
            )
            self.model.load_state_dict(ckpt, strict=False)
            
            print("weigth load")
        
        if num_classes is not None:
            self.model.classifier = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        else:
            self.model.classifier = torch.nn.Identity()
        

    def forward(self, x):
        x = self.model(x) # bs, 1280
        return x


def _strip_backbone_prefix(state_dict):
    """Чекпоинт прошлой стадии → веса для timm-backbone.

    Принимает:
      * сырой timm state_dict (грузим как есть);
      * Lightning-чекпоинт (есть 'state_dict'): срезаем известный префикс backbone.
    Префиксы: новый трейнер → 'model.backbone.'; SSL/BYOL → 'model.backbone.model.';
    DINOv2 → 'model.teacher.backbone.' (берём teacher — EMA, лучше student);
    старый embeddings-трейнер → 'model.model.'.
    """
    sd = state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else state_dict
    prefixes = ("model.teacher.backbone.", "model.student.backbone.",
                "model.backbone.model.", "model.backbone.", "model.model.")
    prefix = next((p for p in prefixes if any(k.startswith(p) for k in sd)), None)
    if prefix is None:
        return sd  # уже «голый» backbone-state_dict
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class ClassificationModel(nn.Module):
    """Универсальный backbone (ЛЮБАЯ timm-модель) + опциональная embedding-голова.

    Контракт: forward(x) -> {"embeddings": (B, D)}. Веса классификатора и функция
    потерь живут в отдельной голове (см. src.losses.heads), а НЕ здесь — это и
    обеспечивает единый трейнер для CE/BCE/AAM/SubCenter и т.д.

    Параметры:
      model_name      — любое имя для timm.create_model ('convnextv2_base.fcmae',
                        'efficientnet_v2_s', 'swin_small_patch4_window7_224', 'vit_base_patch16_224', ...);
      pretrained      — качать предобученные веса timm;
      embedding_dim   — если задан, проекция num_features -> embedding_dim (Linear+BN);
                        иначе эмбеддинги = выход backbone (num_features);
      dropout         — dropout на эмбеддингах;
      backbone_ckpt   — локальный backbone/Lightning-чекпоинт прошлой стадии (strict=False);
      timm_kwargs     — проброс в timm.create_model (drop_path_rate, img_size, ...).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        embedding_dim: int | None = None,
        dropout: float = 0.0,
        backbone_ckpt: str | None = None,
        **timm_kwargs,
    ):
        super().__init__()
        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, **timm_kwargs
        )
        in_features = self.backbone.num_features

        if backbone_ckpt is not None:
            ckpt = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
            sd = _strip_backbone_prefix(ckpt)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[ClassificationModel] backbone load: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")

        if embedding_dim is not None:
            self.embedding_head = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
            self.feature_dim = embedding_dim
        else:
            self.embedding_head = nn.Identity()
            self.feature_dim = in_features

        self.dropout = nn.Dropout(dropout)

    @property
    def num_features(self):
        return self.feature_dim

    def extract_embeddings(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return self.dropout(embeddings)

    def forward(self, x):
        return {"embeddings": self.extract_embeddings(x)}


class SpatialFeatureModel(nn.Module):
    """Backbone, возвращающий КАРТУ признаков ``[B, C, H', W']`` (без global pooling).

    Для голов, которым нужна пространственная сетка признаков (напр. ``SheafADMMHead``),
    а не пулинг-эмбеддинг. Контракт как у ``ClassificationModel``: forward(x) ->
    {"embeddings": feature_map}, но "embeddings" здесь — 4D тензор.

    proj_dim: если задан, 1x1-conv (+BN+GELU) сжимает C -> proj_dim (контроль ёмкости и
              размера входа энкодера пучка). None -> сырые каналы backbone'а.
    Работает с CNN-backbone'ами timm (ConvNeXt/EfficientNetV2/...), где
    global_pool='' даёт (B, C, H, W).
    """

    def __init__(self, model_name, pretrained=True, proj_dim=None,
                 backbone_ckpt=None, **timm_kwargs):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", **timm_kwargs
        )
        in_features = self.backbone.num_features

        if backbone_ckpt is not None:
            ckpt = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
            sd = _strip_backbone_prefix(ckpt)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[SpatialFeatureModel] backbone load: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")

        if proj_dim is not None:
            self.proj = nn.Sequential(
                nn.Conv2d(in_features, proj_dim, kernel_size=1),
                nn.BatchNorm2d(proj_dim),
                nn.GELU(),
            )
            self.out_channels = proj_dim
        else:
            self.proj = nn.Identity()
            self.out_channels = in_features

    def forward(self, x):
        fmap = self.backbone(x)                    # [B, C, H', W']  (CNN, global_pool='')
        if fmap.dim() != 4:
            raise ValueError(
                f"{self.model_name}: ожидалась 4D карта признаков, получено {tuple(fmap.shape)}. "
                "SpatialFeatureModel рассчитан на CNN-backbone (ConvNeXt/EffNetV2)."
            )
        return {"embeddings": self.proj(fmap)}


class TimmModel(nn.Module):
    """Обёртка над timm.create_model (см. https://timm.fast.ai/create_model).

    num_classes == 0 (или None) → feature extractor: forward(x) -> эмбеддинги (B, num_features).
    num_classes  > 0            → классификатор:    forward(x) -> логиты   (B, num_classes).

    path_to_local_weigth: локальный backbone-чекпоинт (.pth в формате timm, strict=False) —
        для дообучения с SSL/stage2 backbone (классификаторная голова инициализируется заново).
    pretrained: качать предобученные веса timm (например 'convnextv2_base.fcmae').
    Доп. kwargs пробрасываются в timm.create_model (drop_path_rate, img_size, ...).
    """

    def __init__(self, model_name, num_classes=0, pretrained=True,
                 path_to_local_weigth=None, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes or 0, **kwargs
        )
        if path_to_local_weigth is not None:
            ckpt = torch.load(path_to_local_weigth, map_location="cpu", weights_only=False)
            #ckpt = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(ckpt, strict=True)
            print("weight load")

    @property
    def num_features(self):
        return self.model.num_features

    def forward(self, x):
        return self.model(x)