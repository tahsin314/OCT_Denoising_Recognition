# OCT Recognition 
This Repo contains my scripts for the *OCT Recognition Task*.

## Features

- &#x2611; Balanced Sampler 

- &#x2611; Mixed Precision

- &#x2611; Gradient Accumulation  

- &#x2611; Optimum Learning Rate Finder [LR Finder Suggestion is terrible. I just observed the learning rate at which loss starts to diverge and set `learning_rate = learning rate at diverging loss/100`. No particular intention behind it.] 

- &#x2611; Out of Fold

## Resources

## Can be useful

## How to run

- Run `git clone https://github.com/tahsin314/OCT_Recognition_`
- Run `conda env create -f environment.yml` and then `conda activate ipcv`
- Run `train.py`. Change parameters according to your preferences from the `oct_config.ini` file before training.
- Run `denoising_ae_train.py` for training the denoising autoencoder.
- `dr_config` parameters:

    ```
    n_fold = Total number of folds
    fold = fold that you want to keep as your validation set
    SEED = Seed value. This value will be use for random initialization
    batch_size 
    sz = Image Dimension
    learning_rate 
    patience = patience for Learning Rate Scheduler
    accum_step = Gradient Accumulation steps 
    num_class = If regression, it should be 1 otherwise 5
    gpu_ids = GPUs to use
    mixed_precision 
    pretrained_model = model name
    model_type = Normal, Variational
    cam_layer_name = Class Activation Mapping Layer
    model_dir = Directory where models will be saved
    distributed_backend = None, dp or ddp. Currently ddp does not work
    mode = train or lr_finder
    load_model = 0 if False else 1
    imagenet_stats = mean and standard deviation of imagenet data
    n_epochs = number of epochs to train
    TTA = Test Time Augmentation
    oof = Out of Fold(1 if True)
    balanced_sampler 
    data_dir 
    image_path 
    test_image_path 
    ```

## Tasks

- Add Kuan Filter
