import time
from typing import Dict

import numpy as np
from chex import Numeric

from environment_wrappers.base import BaseEnvironment
from utils.loggers import BaseLogger
from utils.replay_buffers import Experience, FlashbaxReplayBuffer


class BaseMARLSystem:
    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        discount: float = 0.99,
        add_agent_id_to_obs_in_trainer: bool = True,
        add_agent_id_to_obs_in_action_selection: bool = True
    ):
        self._environment = environment
        self._agents = environment.possible_agents
        self._logger = logger
        self._discount = discount
        self._add_agent_id_to_obs_in_trainer = add_agent_id_to_obs_in_trainer
        self._add_agent_id_to_obs_in_action_selection = add_agent_id_to_obs_in_action_selection

        self._env_step_ctr = 0.0
        self._eval_step_counter = 0.0

    def get_stats(self) -> Dict[str, Numeric]:
        return {}

    def evaluate(self, num_eval_episodes: int = 4) -> Dict[str, Numeric]:
        """Method to evaluate the system online (i.e. in the environment)."""
        episode_returns = []

        for _ in range(num_eval_episodes):
            self.reset()
            observations_ = self._environment.reset()

            if isinstance(observations_, tuple):
                observations, infos = observations_
            else:
                observations = observations_
                infos = {}

            done = False
            episode_return = 0.0
            while not done:
                if "legals" in infos:
                    legal_actions = infos["legals"]
                else:
                    legal_actions = None

                actions = self.select_actions(observations, legal_actions, explore=False)

                observations, rewards, terminals, truncations, infos = self._environment.step(
                    actions
                )

                self._eval_step_counter += 1

                episode_return += np.mean(list(rewards.values()), dtype="float")
                done = all(terminals.values()) or all(truncations.values())
            episode_returns.append(episode_return)
        logs = {"evaluator/episode_return": np.mean(episode_returns)}

        return logs

    def train_offline(
        self,
        replay_buffer: FlashbaxReplayBuffer,
        max_trainer_steps: int = int(1e5),
        evaluate_every: int = 1000,
        num_eval_episodes: int = 4,
    ) -> None:
        """Method to train the system offline.

        WARNING: make sure evaluate_every % log_every == 0 and log_every < evaluate_every,
        else you won't log evaluation.
        """
        trainer_step_ctr = 0
        while trainer_step_ctr < max_trainer_steps:
            if evaluate_every is not None and trainer_step_ctr % evaluate_every == 0:
                print("EVALUATION")
                eval_logs = self.evaluate(num_eval_episodes)
                self._logger.write(
                    eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True
                )

            start_time = time.time()
            experience = replay_buffer.sample()
            end_time = time.time()
            time_to_sample = end_time - start_time

            start_time = time.time()
            train_logs = self.train_step(experience)
            end_time = time.time()
            time_train_step = end_time - start_time

            train_steps_per_second = 1 / (time_train_step + time_to_sample)

            logs = {
                **train_logs,
                "Trainer Steps": trainer_step_ctr,
                "Time to Sample": time_to_sample,
                "Time for Train Step": time_train_step,
                "Train Steps Per Second": train_steps_per_second,
            }

            self._logger.write(logs)

            trainer_step_ctr += 1

        print("FINAL EVALUATION")
        eval_logs = self.evaluate(num_eval_episodes)
        self._logger.write(eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True)
        self._logger.close()

    def reset(self) -> None:
        """Called at the start of each new episode."""
        return

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Dict[str, np.ndarray],
        explore: bool = True,
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def train_step(self, batch: Experience) -> Dict[str, Numeric]:
        raise NotImplementedError
