# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN
import os


class ClsDataConfig:
    def __init__(self, cfg):
        self.data_dir = cfg.DATA_CONF.CLS_DATA_DIR
        self.val_fold = cfg.DATA_CONF.VAL_FOLD
        self.eval_sample_rate = cfg.DATA_CONF.EVAL_SAMPLE_RATE
        self.label_file = cfg.DATA_CONF.LABEL_FILE
        # files                    | sample per category
        # train_k_fold2_c39676.csv | s/c >=10
        # train_k_fold2_c27756.csv | s/c >=15


def add_metric_config(cfg):
    """
    Add config for DETR.
    """
    cfg.MODEL.META_ARCHITECTURE = 'MetricLearningArch'

    cfg.MODEL.METRIC_LEARN = CN()
    cfg.MODEL.METRIC_LEARN.MARGINS = CN()
    cfg.MODEL.METRIC_LEARN.MARGINS.ADAPTIVE = False
    cfg.MODEL.METRIC_LEARN.MARGINS.VALUE = 0.4

    cfg.MODEL.METRIC_LEARN.FEATURE_SCALER = 64.0
    cfg.MODEL.METRIC_LEARN.EASY_MARGIN = True

    cfg.MODEL.METRIC_LEARN.LOSS = CN()
    cfg.MODEL.METRIC_LEARN.LOSS.NAME = 'ArcMarginModelLoss' #'ArcFaceLossAdaptiveMargin'
    cfg.MODEL.METRIC_LEARN.LOSS.FOCAL_LOSS = True
    cfg.MODEL.METRIC_LEARN.LOSS.GAMMA = 2.0

    cfg.MODEL.METRIC_LEARN.NUM_CLASSES = -1
    cfg.MODEL.METRIC_LEARN.SUBCENTRES_K = 3

    cfg.INPUT.IMAGE_SIZE_MSQ = 224
    cfg.MODEL.METRIC_LEARN.EMBEDDING_SIZE = 512
    cfg.MODEL.METRIC_LEARN.BOTTLENECK_TYPE = ''  # ["LBA", "DLBA", "LB"]

    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1


def add_vit_config(cfg):
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.PRE_TRAINED = True
    cfg.MODEL.VIT.DEFAULT_CONF = 'vit_small_patch16_224'
    cfg.MODEL.BACKONE.NAME = 'vit_small_patch16_224'


def add_cls_data_config(cfg):
    cfg.DATA_CONF = CN()
    cfg.DATA_CONF.CLS_DATA_DIR = '/datavol/data'
    cfg.DATA_CONF.EVAL_SAMPLE_RATE = 1/100
    cfg.DATA_CONF.LABEL_FILE = ''

    # Splitted the dataset into 5 fold, specify the
    cfg.DATA_CONF.VAL_FOLD = 0


def add_log_output_config(cfg, project_name):
    cfg.OUTPUT_DIR = '/datavol/cls_log_output{}'.format(project_name)
    cfg.CHECKPOINT_DIR = '/datavol/cls_checkpoints/{}'.format(project_name)
    cfg.TBOARD_DIR = '/datavol/cls_tensorboard/{}'.format(project_name)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.TBOARD_DIR, exist_ok=True)


