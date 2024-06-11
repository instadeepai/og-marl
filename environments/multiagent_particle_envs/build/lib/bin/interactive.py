#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

import numpy as np

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('--num_agents', default=3, type=int)
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    if args.scenario == 'simple_spread.py':
        world = scenario.make_world(num_agents=args.num_agents)
    else:
        world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(
        world, 
        scenario.reset_world, 
        scenario.reward, 
        scenario.observation, 
        info_callback=None, 
        shared_viewer=True, 
        discrete_action=False,
    )
    
    obs_n = env.reset()
    while True:
        start = time.time()
        act_n = []
        for i in range(len(obs_n)):
            curr_ac = 2 * np.random.rand(2) - 1
            act_n.append(curr_ac)
        obs_n, reward_n, done_n, _ = env.step(act_n)
        end = time.time()
        elapsed = end - start
        time.sleep(max(1 / 30 - elapsed, 0))
