from detectron2.modeling import META_ARCH_REGISTRY, BACKBONE_REGISTRY, build_backbone

from detectron2.layers import ShapeSpec
from timm.models import HybridEmbed, PatchEmbed, VisionTransformer as VT, load_pretrained
import torch
import torch.nn as nn

from timm.models.vision_transformer import default_cfgs, _conv_filter


class VisionTransformer(VT):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs:
                img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm
        """
        super().__init__(**kwargs)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # x_cls = self.head(x[:, 0])
        # return x_cls
        return x[:, 0]  # First token embedding just like bert


@BACKBONE_REGISTRY.register()
def vit_small_patch16_224(cfg, input_shape: ShapeSpec, **kwargs):
    if cfg.MODEL.VIT.PRE_TRAINED:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)

    # FIXME: Here we need to convert D2 config to timm config
    model.default_cfg = default_cfgs[cfg.MODEL.VIT.DEFAULT_CONF]
    if cfg.MODEL.VIT.PRE_TRAINED:
        load_pretrained(
            model, num_classes=kwargs.get('num_classes', 0),
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=_conv_filter)
    return model
