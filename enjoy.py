import argparse
import gym
import os
import numpy as np
import random
import tensorflow as tf 
import torch

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.experiments.atari.model import model, dueling_model

from model_attn import *
from collections import deque, namedtuple
Transition = namedtuple("Transition", ["state", "action"])
replay_memory = []
replay_memory_size = 500 * 1000
batch_size = 64
attn_net = Attn()
if torch.cuda.is_available():
    attn_net.cuda()

def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def play(env, act, stochastic, video_path):
    num_episodes = 0
    attn_net_play = False
    obs = env.reset()
    while True:
        action = act(np.array(obs)[None], stochastic=stochastic)[0]

        if len(replay_memory) == replay_memory_size: # pop
            replay_memory.pop(0)
        replay_memory.append(Transition(np.array(obs), action))
        if len(replay_memory) > 1000: # train
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch = map(np.array, zip(*samples))
            print('accuracy %.2f%%' % attn_net.train_(states_batch, action_batch))

        if attn_net_play: # play
            action = attn_net.action_(np.array(obs))
        obs, rew, done, info = env.step(action)
        
        if len(info["rewards"]) > num_episodes:
            print(attn_net_play, info["rewards"][-1])
            num_episodes = len(info["rewards"])
        if done:
            obs = env.reset()
            attn_net_play = np.random.randint(10) == 0


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4,
        gpu_options=gpu_options)

    with tf.Session(config=tf_config) as sess:
        args = parse_args()
        env = make_env(args.env)
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(env, act, args.stochastic, args.video)

