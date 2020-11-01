import logging
import os

import pandas as pd
import torch
from detectron2.data import MapDataset, DatasetCatalog, MetadataCatalog
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler, TrainingSampler, RepeatFactorTrainingSampler
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset

from .config import ClsDataConfig
from .mapper import ClsDatasetMapper
import numpy as np


def build_classification_loader(cfg, dataset_name, mapper=None, is_train=True):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        is_train (bool):
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    df = DatasetCatalog.get(dataset_name)

    dataset = CsvImageDataset(csv_dataframe=df, is_train=is_train)
    if mapper is None:
        mapper = ClsDatasetMapper(cfg, is_train=is_train)
    dataset = MapDataset(dataset, mapper)

    if is_train:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        # TODO avoid if-else?
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            # repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            #     dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            # )
            # sampler = RepeatFactorTrainingSampler(repeat_factors)
            raise NotImplementedError()
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))
        batch_sampler = BatchSampler(sampler, cfg.SOLVER.IMS_PER_BATCH, drop_last=False)
    else:
        sampler = InferenceSampler(len(dataset))
        batch_sampler = BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


class CsvImageDataset(Dataset):
    def __init__(self, csv_dataframe, is_train):
        self.csv_df = csv_dataframe.reset_index()
        self.is_train = is_train

    def __len__(self):
        return self.csv_df.shape[0]

    def __getitem__(self, index):
        row = self.csv_df.iloc[index]

        d = {'fp': row.filepath, 'label': row.landmark_id}
        return d


class CSVDataAPI:
    def __init__(self, data_config: ClsDataConfig):
        self.data_dir = data_config.data_dir
        self.label_file = data_config.label_file
        self.eval_sample_rate = int(1/data_config.eval_sample_rate)
        self.df, self.num_classes = self.get_df(sort_by_label=True)
        tmp = np.sqrt(1 / np.sqrt(self.df['landmark_id'].value_counts().sort_index().values))
        self.margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05
        return

    def get_df(self, sort_by_label=True):
        data_dir = self.data_dir
        df = pd.read_csv(os.path.join(data_dir, self.label_file))

        if sort_by_label == 0:
            pass
        else:
            df = df.sort_values(by=['landmark_id'])
            df = df.reset_index(drop=True)

        df['filepath'] = df['id']. \
            apply(lambda x: os.path.join(data_dir, 'train', x[0], x[1], x[2], f'{x}.jpg'))

        landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
        idx2landmark_id = {idx: landmark_id for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
        df['landmark_id'] = df['landmark_id'].map(landmark_id2idx)

        out_dim = df.landmark_id.nunique()
        return df, out_dim

    def get_train_and_valid_split(self, fold):
        df = self.df
        df_train = df[df['fold'] != fold]
        df_eval_full = df[df['fold'] == fold]
        df_eval = df[df['fold'] == fold].reset_index(drop=True).query(f"index % {self.eval_sample_rate}==0")
        print("========DATASET Brief==========")
        print(f'TRAIN: {len(df_train.index)}, EVAL: {len(df_eval.index)}, EVAL_SAMPLE_RATE: 1/{self.eval_sample_rate}')
        return {'TRAIN': df_train, 'EVAL': df_eval, 'EVAL_FULL': df_eval_full}


def register_lm_cls_datasets(data_config: ClsDataConfig):
    """
    register the classification dataset
    """
    # split_names = ['train', 'eval']
    csv_data_api = CSVDataAPI(data_config=data_config)
    num_classes = csv_data_api.num_classes
    adaptive_margins = csv_data_api.margins
    splits = csv_data_api.get_train_and_valid_split(fold=data_config.val_fold)

    for _name, df in splits.items():
        print("Register annotation file:", _name)
        DatasetCatalog.register(
            name=_name,
            func=lambda d=df: d)
        MetadataCatalog.get(_name).evaluator_type = 'cls_gap'

    return num_classes, adaptive_margins
