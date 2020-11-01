import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import os

from tqdm import tqdm


def process_train_df(train_csv='train.csv', data_root='/datavol/data', fold_k=5, keep_thres=15):
    """

    Args:
        train_csv: kaggle train.csv
        data_root: dir containing train.csv
        fold_k: num of folds K in cross validation split
        keep_thres: only when num of samples greater than fold_k it will produce

    Returns:

    """
    train_k_fold_dump = 'train_k_fold2_c{}_ts{}.csv'
    idx2landmark_id = 'idx2landmark_id.pkl'
    if data_root:
        train_csv = os.path.join(data_root, train_csv)
        train_k_fold_dump = os.path.join(data_root, train_k_fold_dump)
        idx2landmark_id = os.path.join(data_root, 'idx2landmark_id.pkl')

    df_train = pd.read_csv(train_csv)
    cat_dist = df_train.groupby(['landmark_id']).count()
    majority_cats = cat_dist[cat_dist['id'] >= keep_thres]
    df_train = df_train[df_train['landmark_id'].isin(majority_cats.index)]
    df_train = df_train.reset_index(drop=True)

    num_cats = len(majority_cats.index)
    num_samples = len(df_train.index)
    train_k_fold_dump = train_k_fold_dump.format(num_cats, num_samples)

    skf = StratifiedKFold(fold_k, shuffle=True, random_state=233)

    df_train['fold'] = -1
    for i, (train_idx_list, valid_idx_list) in tqdm(enumerate(skf.split(df_train, df_train['landmark_id']))):
        df_train.loc[valid_idx_list, 'fold'] = i

    print('dumping to {}\n num_classes: {}'.format(train_k_fold_dump, len(df_train['landmark_id'].unique())))
    print('Total samples Train & Eval: {}'.format(num_samples))
    df_train.to_csv(train_k_fold_dump, index=False)

    landmark_id2idx = {idx: landmark_id for idx, landmark_id in enumerate(sorted(df_train['landmark_id'].unique()))}
    with open(idx2landmark_id, 'wb') as fp:
        pickle.dump(landmark_id2idx, fp)

    return


if __name__ == '__main__':
    process_train_df()

    # main()
