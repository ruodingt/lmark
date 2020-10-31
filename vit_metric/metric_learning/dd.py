import pandas as pd
import os


def get_df(data_dir='/datavol/data', sort_by_label=True):
    df = pd.read_csv(os.path.join(data_dir, 'train_k_fold.csv'))

    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    if sort_by_label == 0:
        pass
    else:
        cls_81313 = df.landmark_id.unique()
        _dfr_tmp = df_train.set_index('landmark_id')
        _dfr_tmp = _dfr_tmp.loc[cls_81313]
        df_train = _dfr_tmp.reset_index()

    df_train['filepath'] = df_train['id'].\
        apply(lambda x: os.path.join(data_dir, 'train', x[0], x[1], x[2], f'{x}.jpg'))
    df = df_train.merge(df, on=['id', 'landmark_id'], how='left')

    landmark_id2idx = {landmark_id: idx for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    idx2landmark_id = {idx: landmark_id for idx, landmark_id in enumerate(sorted(df['landmark_id'].unique()))}
    df['landmark_id'] = df['landmark_id'].map(landmark_id2idx)

    out_dim = df.landmark_id.nunique()
    return df, out_dim

get_df(sort_by_label=False)
