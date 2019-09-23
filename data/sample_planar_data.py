import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import os
from os import path
from tqdm import trange
import json
from datetime import datetime
import argparse
from PIL import Image

u_dim = 2 # control (action) dimension
width, height = 40, 40
x_dim = width*height
rw = 1 # robot half-width
max_dist = 2
env_file = '/home/tungnd/Desktop/E2C/data/env.npy'
env = np.load(env_file)

def is_colliding(p):
    """
    :param p: the coordinate (x, y) of the agent center
    :return: if agent body overlap with obstacle field
    """
    if np.any([p - rw < 0, p + rw >= height]):  # check if the agent body is out of the plane
        return True

    return np.mean(env[p[0] - rw:p[0] + rw + 1, p[1] - rw:p[1] + rw + 1]) > 0.05

def render(p):
    # return image X given true state p (position) of robot
    x = np.copy(env)
    x[p[0] - rw:p[0] + rw + 1, p[1] - rw:p[1] + rw + 1] = 1.  # robot is white on black background
    return x

def sample_for_eval(num_eval):
    samples = []

    for i in range(num_eval):
        while True:
            row = randint(0 + rw, height - rw)
            col = randint(0 + rw, width - rw)
            if not is_colliding(np.array([row, col])):
                break

        state = np.array([row, col])

        before = render(state)

        # sample action
        u = np.array([0, 0])
        # row direction
        d = randint(-1, 2)  # direction
        nsteps = randint(max_dist + 1)  # step length
        for _ in range(nsteps):
            state[0] += d
            u[0] += d
            if is_colliding(state):
                state[0] -= d
                u[0] -= d
                break

        # column direction
        d = randint(-1, 2)
        nsteps = randint(max_dist + 1)
        for _ in range(nsteps):
            state[1] += d
            u[1] += d
            if is_colliding(state):
                state[1] -= d
                u[1] -= d
                break

        # apply control
        after = render(state)
        samples.append((before, u, after))
    return samples

def sample_traj(len_traj = 5):
    # initial state
    while True:
        row = randint(0 + rw, height - rw)
        col = randint(0 + rw, width - rw)
        if not is_colliding(np.array([row, col])):
            break
    states = [np.array([row, col])]
    actions = []
    for i in range(1, len_traj + 1):
        dr = randint(-1, 2) * randint(max_dist + 1)
        dc = randint(-1, 2) * randint(max_dist + 1)
        u = np.array([dr, dc])
        next_state = states[i-1] + u
        states.append(next_state)
        actions.append(u)
    return states, actions

def sample_planar(sample_size, output_dir = './data/planar'):
    """
    :param sample_size:
    :param output_dir:
    :param max_dist: max distance taken by the agent
    :return: write samples [x_t, u_t, x_{t+1} to file]
    """
    samples = []

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    for i in trange(sample_size):
        """
        for each sample:
        - draw a random state (row, col)
        - render x_t
        - draw a random action u_t and apply
        - render x_t+1 after applying u_t
        """
        # sample until we get a valid position (not colliding)
        while True:
            row = randint(0 + rw, height - rw)
            col = randint(0 + rw, width - rw)
            if not is_colliding(np.array([row, col])):
                break

        state = np.array([row, col])

        initial_state = np.copy(state)
        before = render(initial_state)

        # sample action
        u = np.array([0, 0])
        # row direction
        d = randint(-1, 2)  # direction
        nsteps = randint(max_dist + 1)  # step length
        for _ in range(nsteps):
            state[0] += d
            u[0] += d
            if is_colliding(state):
                state[0] -= d
                u[0] -= d
                break

        # column direction
        d = randint(-1, 2)
        nsteps = randint(max_dist + 1)
        for _ in range(nsteps):
            state[1] += d
            u[1] += d
            if is_colliding(state):
                state[1] -= d
                u[1] -= d
                break

        # apply control
        after_state = state
        after = render(after_state)

        before_file = 'before-{:05d}.png'.format(i)
        Image.fromarray(before * 255.).convert('L').save(path.join(output_dir, before_file))

        after_file = 'after-{:05d}.png'.format(i)
        Image.fromarray(after * 255.).convert('L').save(path.join(output_dir, after_file))

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': u.tolist(),
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': sample_size,
                    'max_distance': max_dist,
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

def main(args):
    sample_size = args.sample_size

    sample_planar(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')

    args = parser.parse_args()

    main(args)