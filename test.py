import os
from config import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold


dirname = '../../data/HARTS/Classes_updated/'

def get_data(dirname, sep='Classes_updated', n_fold=5, random_state=42):
    
    paths = []
    classname = []
    train_idx = []
    val_idx = []
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
    for root, dirs, files in os.walk(dirname, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
            classname.append(int(path.split(f'/{sep}/')[-1].split('/')[0]))
    df = pd.DataFrame(list(zip(paths, classname)), columns=['path', 'classname'])
    for i, (train_index, val_index) in enumerate(skf.split(paths, classname)):
        train_idx = train_index
        val_idx = val_index
        df.loc[val_idx, 'fold'] = i

    df['fold'] = df['fold'].astype('int')

    return df

print(get_data(dirname))