import os
from tqdm.auto import tqdm as T
import pandas as pd
from sklearn.model_selection import StratifiedKFold


dirname = '../../data/OCT/oct2017/OCT2017 /'
SEED = 42

def nurun(image_name):
    for i in range(25):
        for j in range(10):
            os.system(f'mv Page{i*10+j+1}.jpg {i}_{j+1}.jpg')

def get_data(dirname, n_fold=5, random_state=42):
    
    paths = []
    classname = []
    train_idx = []
    val_idx = []
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)
    for root, dirs, files in T(os.walk(dirname, topdown=False)):
        # print(root, dirs)
        for name in files:
            path = os.path.join(root, name)
            if 'jpeg' in path:
                paths.append(path)
                classname.append(path.split('/')[-2])
    classes = list(set(classname))
    class_id = {c: i for i, c in enumerate(classes)}
    df = pd.DataFrame(list(zip(paths, classname)), columns=['id', 'classname'])
    for i, (train_index, val_index) in enumerate(skf.split(paths, classname)):
        train_idx = train_index
        val_idx = val_index
        df.loc[val_idx, 'fold'] = i
    df['target'] = df['classname'].apply(lambda x: class_id[x])
    df['fold'] = df['fold'].astype('int')

    return df, class_id

df = get_data(dirname)
print(df.head(20))
df.to_csv(f'{dirname}/OCT2017_fold.csv', index=False)