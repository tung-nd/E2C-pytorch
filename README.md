## Embed to Control

This is a pytorch implementation of the paper "Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images", NIPS, 2015.

**Note: This is not and official implementation.**

### Installing

First, clone the repository:

```
git clone https://github.com/tungnd1705/E2C-pytorch.git
```

Install the dependencies as listed in `env.yml` and activate the environment

```
conda env create -f env.yml

conda activate e2c
```

Then install the patch version of gym in order to sample the pendulum data

```
cd gym

python setup.py install
```

### Simulate training data

Currently the code supports simulating 3 environments: `planar`, `pendulum` and `cartpole`.

In order to generate data, simply run `python sample_{env_name}_data.py --sample_size={sample_size}`.

**Note: the sample size is equal to the total number of training and test data**

For the planar task, we base on [this](https://github.com/ethanluoyc/e2c-pytorch) implementation and modify for our needs.

### Training

Run the ``train_e2c.py`` with your own settings. Example:

```
python train_e2c.py \
    --env=planar \
    --propor=3/4 \
    --batch_size=128 \
    --lr=0.0001 \
    --lam=0.25 \
    --num_iter=5000 \
    --iter_save=1000
```

You can visualize the training process by running ``tensorboard --logdir=logs``.

### Citation

If you find E2C useful in your research, please consider citing:

```
@inproceedings{watter2015embed,
  title={Embed to control: A locally linear latent dynamics model for control from raw images},
  author={Watter, Manuel and Springenberg, Jost and Boedecker, Joschka and Riedmiller, Martin},
  booktitle={Advances in neural information processing systems},
  pages={2746--2754},
  year={2015}
}
```