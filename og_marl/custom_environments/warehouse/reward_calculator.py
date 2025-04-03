import math
from statistics import mean

from .observation_parser import NearInformation, Observation, ObservationParser


class RewardCalculator:

    @staticmethod
    def position_reward(env, x, y):
        max_dist = []
        dist = []

        for goal in env.goals:
            goal_x = goal[0]
            goal_y = goal[1]
            dist.append(math.hypot(goal_x - x, goal_y - y))
            max_dist.append(math.hypot(goal_x - 0, goal_y - 0))
        return 0.0005 * (mean(max_dist) - mean(dist)) / mean(max_dist)

    @staticmethod
    def is_center_shelf(obs: Observation, is_requested: bool) -> bool:
        center_info: NearInformation = obs.near_info[4]
        return center_info.is_shelf and center_info.is_requested_shelf

    @staticmethod
    def find_requested_shelf(obs: Observation) -> NearInformation:
        near_info = obs.near_info
        info: NearInformation
        for info in near_info:
            if info.is_shelf and info.is_requested_shelf:
                return info
        return None

    @staticmethod
    def calculate(env, reward, prev_obs, obs):
        obs: Observation = ObservationParser.parse(obs)

        # requested shelf
        if RewardCalculator.is_center_shelf(obs, True):
            if obs.is_carrying:
                reward += 0.006
            else:
                reward += 0.003

            # reward += RewardCalculator.position_reward(env, obs.x, obs.y)
        #
        # # non requested shelf
        # if RewardCalculator.is_center_shelf(obs, False):
        #     if obs.is_carrying:
        #         reward -= 0.003
        #     else:
        #         reward -= 0.0015
        #
        #     reward -= RewardCalculator.position_reward(env, obs.x, obs.y)

        # find out requested item
        if RewardCalculator.find_requested_shelf(obs) is not None:
            reward += 0.001

        return reward
