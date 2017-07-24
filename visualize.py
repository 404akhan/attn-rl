import argparse
import gym
import os
import numpy as np
import random
import tensorflow as tf 

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.experiments.atari.model import model, dueling_model

from visualize_model import *
from collections import deque, namedtuple
Transition = namedtuple("Transition", ["state", "action"])
replay_memory = []
replay_memory_size = 500 * 1000
upd_init_size = 10 * 1000
batch_size = 16
attn_net = Attn()
if attn_net.cuda_exist:
    attn_net.cuda()

filename = 'model-torch/counter_31000.pth'
print('==> loading checkpoint {}'.format(filename))
checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
attn_net.load_state_dict(checkpoint)
print('==> loaded checkpoint {}'.format(filename))

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
    attn_net_play = False
    counter = 0
    counter_games = 0
    reward_sum = 0
    obs = env.reset()
    accuracy_arr = []
    counter_frame = 0

    while True:
        counter_frame += 1
        action = act(np.array(obs)[None], stochastic=stochastic)[0]

        if len(replay_memory) == replay_memory_size: # pop
            replay_memory.pop(0)
        replay_memory.append(Transition(np.array(obs), action))

        if len(replay_memory) > batch_size and counter_frame > 100: # train
            counter += 1
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch = map(np.array, zip(*samples))
            acc = attn_net.train_(states_batch, action_batch)
            accuracy_arr.append(acc)

            if counter % 100 == 0:
                print('update %d, accuracy %.2f%%' % (counter, np.mean(accuracy_arr)))
                accuracy_arr = []
                if counter % 1000 == 0: attn_net.save_model(counter)

        if attn_net_play: # play
            action = attn_net.action_(np.array(obs))
        obs, rew, done, info = env.step(action)
        reward_sum += rew
        
        if done:
            counter_games += 1
            obs = env.reset()
            print(attn_net_play, reward_sum)
            
            # attn_net_play = counter_games%10 == 0
            reward_sum = 0
            counter_frame = 0


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=8,
        intra_op_parallelism_threads=8,
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

