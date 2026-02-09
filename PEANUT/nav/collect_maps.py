import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt

from agent.peanut_agent import PEANUT_Agent
from nav.patch_config import patch_task_config

# HARDCODE FOR NOW - UPDATE AS YOU NEED
os.environ["CHALLENGE_CONFIG_FILE"] = "path/to/PEANUT/configs/challenge_objectnav2022.local.rgbd.yaml"

def shuffle_episodes(env, shuffle_interval):
    ranges = np.arange(0, len(env.episodes), shuffle_interval)
    np.random.shuffle(ranges)
    new_episodes = []
    for r in ranges:
        new_episodes += env.episodes[r:r + shuffle_interval]
    env.episodes = new_episodes
    
def main():

    args_2 = get_args()
    args_2.only_explore = 1  
    args_2.switch_step = 999
    args_2.global_downscaling = 4
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    patch_task_config(config)
    print(config.habitat.dataset.split)
    
    nav_agent = PEANUT_Agent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    print(hab_env.episode_iterator._max_rep_step)
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 20 * 50
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes
    
    save_steps = list(range(5, 525, 20))
    succs, spls, dtgs, epls = [], [], [], []
    
    count_episodes = 0
    while count_episodes < min(num_episodes, end):
        if count_episodes < 384:
            count_episodes += 1
            continue

        observations = hab_env.reset()
        observations['objectgoal'] = [0]
        nav_agent.reset()
        print(count_episodes, hab_env._current_episode.scene_id)
        sys.stdout.flush()
        
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            full_map_seq = np.zeros((len(save_steps), 4 + args_2.num_sem_categories, 
                                     nav_agent.agent_states.full_w, nav_agent.agent_states.full_h), dtype=np.uint8)
            while not hab_env.episode_over:
                sys.stdout.flush()
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                observations['objectgoal'] = [0]
                          
                if step_i % 100 == 0:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                if step_i in save_steps:
                    full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                    full_map_seq[seq_i] = full_map.astype(np.uint8)
                    seq_i += 1
                    
                
            if np.sum(full_map_seq[:, 4:]) > 0 and np.sum(full_map_seq[:, 1]) > 4000:
                np.savez_compressed('./data/saved_maps/%s_80/f%05d.npz' % (config.habitat.dataset.split, count_episodes), maps=full_map_seq)

        count_episodes += 1
        

if __name__ == "__main__":
    main()
