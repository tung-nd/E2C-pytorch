import os
from os import path
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import gym
import json
from datetime import datetime
import argparse

def sample_cartpole(sample_size, output_dir='data/cartpole', step_size=5, apply_control=True, num_shards=10):
    env = gym.make('CartPole-v0').env
    assert sample_size % num_shards == 0

    samples = []

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    state = env.reset()
    for i in trange(sample_size):
        """
        for each sample:
        - draw a random state
        - render x_t
        - draw a random action u_t and apply
        - render x_t+1 after applying u_t
        """
        # state = env.reset()

        initial_state = state
        before = env.render(mode = 'rgb_array')

        # apply the same control over a few timesteps
        if apply_control:
            u = env.action_space.sample()
        else:
            u = np.zeros((1,))

        for _ in range(step_size):
            state, reward, done, info = env.step(u)

        after_state = state
        after = env.render(mode = 'rgb_array')

        shard_no = i // (sample_size // num_shards)

        shard_path = path.join('{:03d}-of-{:03d}'.format(shard_no, num_shards))

        if not path.exists(path.join(output_dir, shard_path)):
            os.makedirs(path.join(output_dir, shard_path))

        before_file = path.join(shard_path, 'before-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, before_file), before)

        after_file = path.join(shard_path, 'after-{:05d}.jpg'.format(i))
        plt.imsave(path.join(output_dir, after_file), after)

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': [u],
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'step_size': step_size,
                    'apply_control': apply_control,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

    env.viewer.close()

def main(args):
    sample_size = args.sample_size

    sample_cartpole(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)