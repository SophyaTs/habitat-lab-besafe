import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
from habitat.sims.unreal.statistics import Statistics
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt

from agent.peanut_agent import PEANUT_Agent
from nav.patch_config import patch_task_config

# HARDCODE FOR NOW
os.environ["CHALLENGE_CONFIG_FILE"] = "/mnt/storage/University/ETH/Thesis/PEANUT/configs/challenge_objectnav2022.local.rgbd.yaml"


def main():

    args = get_args()
    args.only_explore = 0  
    
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    patch_task_config(config)
    
    hab_env = Env(config=config)
    nav_agent = PEANUT_Agent(args=args,task_config=config)
    print(config.habitat.dataset.split, 'split')
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 1e4
    start = args.start_ep
    end = args.end_ep if args.end_ep > 0 else num_episodes
    
    sucs, spls, ep_lens = [], [], []
    
    ep_i = 0
    offset =  0
    Statistics.set_offset(offset)
    while ep_i < min(num_episodes, end):
        if ep_i < offset:
        # if hab_env._current_episode.object_category != "chair":
            hab_env._current_episode = next(hab_env.episode_iterator)
            ep_i +=1 
            continue

        observations = hab_env.reset()
        nav_agent.reset()
        print('-' * 40)
        sys.stdout.flush()
             
        if ep_i >= start and ep_i < end:
            print('Episode %d | Target: %s' % (ep_i, hm3d_names[observations['objectgoal'][0]]))
            print('Scene: %s' % hab_env._current_episode.scene_id)

            step_i = 0
            seq_i = 0
            
            while not hab_env.episode_over:
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                          
                if step_i % 100 == 0:
                    print('step %d...' % step_i)
                    sys.stdout.flush()

                step_i += 1
                    
            if args.only_explore == 0:
                
                print('ended at step %d' % step_i)
                
                # Navigation metrics
                metrics = hab_env.get_metrics()
                print(metrics)
                
                # Log the metrics (save them however you want)
                sucs.append(metrics['success'])
                Statistics.submit_success(metrics['success'])
                # spls.append(metrics['spl'])
                ep_lens.append(step_i)
                print('-' * 40)
                print('Average Success: %.4f' % (np.mean(sucs))) #, Average SPL: %.4f,np.mean(spls)))
                print('-' * 40)
                sys.stdout.flush()
                
        ep_i += 1
    Statistics.finalize_episode(True)
        

if __name__ == "__main__":
    main()
