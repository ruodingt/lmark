from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
import torch
import torch.nn as nn
from detectron2.structures import ImageList
import re
from .layers import ArcSubCentre
import numpy as np


@META_ARCH_REGISTRY.register()
class MetricLearningArch(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.vis_period = cfg.VIS_PERIOD

        # FIXME: We need to confirm this FORMAT for VIT for D2 backbone it is normal BGR
        self.input_format = cfg.INPUT.FORMAT

        self.is_conv_backbone = re.search('res', cfg.MODEL.BACKBONE.NAME) is not None

        if self.is_conv_backbone:
            # Wemight have non-CONV backbone like VIT
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            dim_backbone_feature_dim = 2048  # for residual net
        else:
            raise NotImplementedError()
        self.cosine_similarity_to_sub_centers = ArcSubCentre(
            in_features=dim_backbone_feature_dim,
            out_features=cfg.MODEL.METRIC_LEARN.NUM_CLASSES,
            k=cfg.MODEL.METRIC_LEARN.SUBCENTRES_K,
            embedding_size=cfg.MODEL.METRIC_LEARN.EMBEDDING_SIZE,
            neck_type=cfg.MODEL.METRIC_LEARN.BOTTLENECK_TYPE)

        if isinstance(cfg.MODEL.METRIC_LEARN.MARGINS.VALUE, list):
            margin = torch.tensor(cfg.MODEL.METRIC_LEARN.MARGINS.VALUE)
        else:
            margin = cfg.MODEL.METRIC_LEARN.MARGINS.VALUE

        if cfg.MODEL.METRIC_LEARN.LOSS.NAME == 'ArcFaceLossAdaptiveMargin':
            raise NotImplementedError()
            # self.loss_fn = ArcFaceLossAdaptiveMargin(margins=margin, s=80)
        elif cfg.MODEL.METRIC_LEARN.LOSS.NAME == 'ArcMarginModelLoss':
            from .layers import ArcMarginModelLoss
            self.loss_fn = ArcMarginModelLoss(margin_m=margin,
                                              feature_scaler=cfg.MODEL.METRIC_LEARN.FEATURE_SCALER,
                                              easy_margin=cfg.MODEL.METRIC_LEARN.EASY_MARGIN,
                                              focal_loss=cfg.MODEL.METRIC_LEARN.LOSS.FOCAL_LOSS,
                                              gamma=cfg.MODEL.METRIC_LEARN.LOSS.GAMMA)
        else:
            print("{} is not yet implemented".format(cfg.MODEL.METRIC_LEARN.LOSS.NAME))
            raise NotImplementedError()

        self.num_classes = cfg.MODEL.METRIC_LEARN.NUM_CLASSES

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        labels = [x["label"].to(self.device) for x in batched_inputs]
        labels = torch.stack(tensors=labels)
        return images, labels

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        images, labels = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        inference_out = self.inference(features['res5'])

        losses = {}
        loss = self.loss_fn.forward(inference_out['classwise_cos_similarities'], labels=labels)
        losses['loss_arc_margin'] = loss

        if not self.training:
            eval_inference_out = {}
            eval_inference_out.update(losses)
            eval_inference_out.update(inference_out)
            return eval_inference_out

        return losses

    def inference(self, features):
        if self.is_conv_backbone:
            x = self.avgpool(features)
            x = torch.flatten(x, 1)
        else:
            x = features

        classwise_cos_similarities = self.cosine_similarity_to_sub_centers(x)
        out_max = classwise_cos_similarities.max(1)
        similarity = out_max.values
        label = out_max.indices

        return {'label': label,
                'similarity': similarity,
                'classwise_cos_similarities': classwise_cos_similarities}
