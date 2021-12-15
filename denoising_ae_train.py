import os
import glob
from functools import partial
import gc
from matplotlib.pyplot import axis
from torch import nn
from config import *
import shutil
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm import tqdm as T
from sklearn.model_selection import StratifiedKFold

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, 
LearningRateMonitor, StochasticWeightAveraging,) 
from pytorch_lightning.loggers import WandbLogger
from OCTDataset import OCTDataset, OCTDataModule
from catalyst.data.sampler import BalanceClassSampler
from losses.psnr import PSNR
from losses.dice import DiceLoss
from utils import *
from model.unet import resUne_t
from OCTModule import LightningOCT
import wandb

seed_everything(SEED)
os.system("rm -rf *.png *.csv")
if mode == 'lr_finder':
  wandb.init(mode="disabled")
  wandb_logger = WandbLogger(project="OCT_Denoising", config=params, settings=wandb.Settings(start_method='fork'))
else:
  wandb_logger = WandbLogger(project="OCT_Denoising", config=params, settings=wandb.Settings(start_method='fork'))
  wandb.init(project="OCT", config=params, settings=wandb.Settings(start_method='fork'))
  wandb.run.name= model_name
labels = [i for i in class_id.keys()]
optimizer = optim.AdamW
# base_criterion = PSNR()
# base_criterion = nn.MSELoss(reduction='sum')
base_criterion = DiceLoss()
criterions = [base_criterion]
# criterion = criterion_margin_focal_binary_cross_entropy

lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

for f in range(n_fold):
    print(f"FOLD #{f}")
    # train_df = df[(df['fold']!=f)]
    # valid_df = df[df['fold']==f]
    
    base = resUne_t(pretrained_model)
    
    wandb.watch(base)
    plist = [ 
        {'params': base.resne_t.parameters(),  'lr': learning_rate/5},
        {'params': base.decoder.parameters(),  'lr': learning_rate}
    ]
    
    train_ds = OCTDataset(train_df.id.values, train_df.target.values, dim=sz, num_class=num_class,
    transforms=train_aug_seg)

    valid_ds = OCTDataset(valid_df.id.values, valid_df.target.values, dim=sz, num_class=num_class, 
    transforms=val_aug_seg)

    test_ds = OCTDataset(test_df.id.values, test_df.target.values, dim=sz,num_class=num_class, 
    transforms=val_aug_seg)
    data_module = OCTDataModule(train_ds, valid_ds, test_ds,  sampler= sampler, 
    batch_size=batch_size)
    cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer(plist, 
    lr=learning_rate), 
    5*len(data_module.train_dataloader()), 1, learning_rate/5, -1)
    
    if mode == 'lr_finder': cyclic_scheduler = None
    model = LightningOCT(model=base, choice_weights=choice_weights, loss_fns=criterions,
    optim= optimizer, plist=plist, batch_size=batch_size, 
    lr_scheduler= lr_reduce_scheduler, num_class=num_class, fold=f, cyclic_scheduler=cyclic_scheduler, 
    learning_rate = learning_rate, random_id=random_id, labels=
    labels, unet=True)
    checkpoint_callback1 = ModelCheckpoint(
        monitor=f'val_loss_fold_{f}',
        dirpath='model_dir',
        filename=f"{model_name}_loss_fold_{f}",
        save_top_k=1,
        mode='min',
    )

    # checkpoint_callback2 = ModelCheckpoint(
    #     monitor=f'val_micro_f_fold_{f}',
    #     dirpath='model_dir',
    #     filename=f"{model_name}_micro_f_fold_{f}",
    #     save_top_k=1,
    #     mode='max',
    # )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    swa_callback = StochasticWeightAveraging()

    trainer = pl.Trainer(max_epochs=n_epochs, precision=16, 
                      # auto_lr_find=True,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                      gradient_clip_val=100,
                      num_sanity_val_steps=10,
                      profiler="simple",
                      weights_summary='top',
                      accumulate_grad_batches = accum_step,
                      logger=[wandb_logger], 
                      checkpoint_callback=True,
                      gpus=gpu_ids, num_processes=4*len(gpu_ids),
                      # stochastic_weight_avg=True,
                      # auto_scale_batch_size='power',
                      benchmark=True,
                      distributed_backend=distributed_backend,
                      # plugins='deepspeed', # Not working 
                      # early_stop_callback=False,
                      progress_bar_refresh_rate=1, 
                      callbacks=[checkpoint_callback1,
                      lr_monitor])

    if mode == 'lr_finder':
      model.choice_weights = [1.0, 0.0]
      trainer.train_dataloader = data_module.train_dataloader
      # Run learning rate finder
      lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader(), min_lr=1e-6, 
      max_lr=500, num_training=2000)
      # Plot with
      fig = lr_finder.plot(suggest=True, show=True)
      fig.savefig('lr_finder.png')
      fig.show()
    # Pick point based on plot, or get suggestion
      new_lr = lr_finder.suggestion()
      print(f"Suggested LR: {new_lr}")
      exit()

    wandb.log(params)
    trainer.fit(model, datamodule=data_module)
    print(gc.collect())
    try:
      print(f"FOLD: {f} \
        Best Model path: {checkpoint_callback1.best_model_path} Best Score: {checkpoint_callback1.best_model_score:.4f}")
    except:
      pass
    chk_path = checkpoint_callback1.best_model_path
    # chk_path = '/home/UFAD/m.tahsinmostafiz/Playground/OCT_Denoising_Recognition/model_dir/Normal_resnet18d_micro_f_fold_0-v6.ckpt'
    model2 = LightningOCT.load_from_checkpoint(chk_path, model=base, choice_weights=[1.0, 0.0], loss_fns=criterions, optim=optimizer,
    plist=plist, batch_size=batch_size, 
    lr_scheduler=lr_reduce_scheduler, cyclic_scheduler=cyclic_scheduler, 
    num_class=num_class, learning_rate = learning_rate, fold=f, random_id=random_id, labels=labels)

    # trainer.test(model=model2, test_dataloaders=data_module.val_dataloader())
    trainer.test(model=model2, test_dataloaders=data_module.test_dataloader())

    if not oof:
      break

# oof_df = pd.concat([pd.read_csv(fname) for fname in glob.glob('oof_*.csv')])
# oof_df.to_csv(f'oof.csv', index=False)