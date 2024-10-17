# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Dict, Optional, Tuple, List

import numpy as np
from chex import Numeric

from og_marl.environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger, JsonWriter
from og_marl.replay_buffers import Experience, FlashbaxReplayBuffer


class BaseMARLSystem:
    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        discount: float = 0.99,
        add_agent_id_to_obs: bool = False,
    ):
        self._environment = environment
        self._agents = environment.possible_agents
        self._logger = logger
        self._discount = discount
        self._add_agent_id_to_obs = add_agent_id_to_obs

        self._env_step_ctr = 0.0

        self.counter = 0

    def get_stats(self) -> Dict[str, Numeric]:
        return {}

    def evaluate(self, num_eval_episodes: int = 4) -> Tuple[Dict[str, Numeric], list]:
        """Method to evaluate the system online (i.e. in the environment)."""
        episode_returns = []
        all_frames = []
        for _ in range(num_eval_episodes):
            ep_frames = []
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
                ep_frames.append(self._environment.render())

                if "legals" in infos:
                    legal_actions = infos["legals"]
                else:
                    legal_actions = None

                actions = self.select_actions(observations, legal_actions, explore=False)

                observations, rewards, terminals, truncations, infos = self._environment.step(
                    actions
                )
                episode_return += np.mean(list(rewards.values()), dtype="float")
                done = all(terminals.values()) or all(truncations.values())
            episode_returns.append(episode_return)
            all_frames.append(ep_frames)
        logs = {"evaluator/episode_return": np.mean(episode_returns)}
        return logs, all_frames

    def train_online(
        self,
        replay_buffer: FlashbaxReplayBuffer,
        max_env_steps: int = int(1e6),
        train_period: int = 20,
    ) -> None:
        """Method to train the system online."""
        episode_returns = []
        episodes = 0
        while True:  # breaks out when env_steps > max_env_steps
            self.reset()  # reset the system
            observations_ = self._environment.reset()

            if isinstance(observations_, tuple):
                observations, infos = observations_
            else:
                observations = observations_
                infos = {}
 
            episode_return = 0.0
            while True:
                if "legals" in infos:
                    legal_actions = infos["legals"]
                else:
                    legal_actions = None

                start_time = time.time()
                actions = self.select_actions(observations, legal_actions)
                end_time = time.time()
                time_for_action_selection = end_time - start_time

                start_time = time.time()
                (
                    next_observations,
                    rewards,
                    terminals,
                    truncations,
                    next_infos,
                ) = self._environment.step(actions)
                end_time = time.time()
                time_to_step = end_time - start_time

                # Add step to replay buffer
                start_time = time.time()
                replay_buffer.add(observations, actions, rewards, terminals, truncations, infos, self._discount)
                end_time = time.time()
                time_to_add_to_replay = end_time - start_time

                # Critical!!
                observations = next_observations
                infos = next_infos

                # Bookkeeping
                episode_return += np.mean(list(rewards.values()), dtype="float")
                self._env_step_ctr += 1

                if (
                    self._env_step_ctr > 1000 and self._env_step_ctr % train_period == 0
                ):  # TODO burn in period should be a variable
                    # Sample replay buffer
                    start_time = time.time()
                    experience = replay_buffer.sample()
                    end_time = time.time()
                    time_to_sample = end_time - start_time

                    # Train step
                    start_time = time.time()
                    self.counter += 1
                    #self._optimizer.learning_rate = 0.000001 + 0.000999*self._env_step_ctr/max_env_steps
                    #self._cql_weight.assign(1 - self._env_step_ctr/max_env_steps)
                    #print(self._cql_weight)
                    train_logs = self.train_step(experience)
                    end_time = time.time()
                    time_train_step = end_time - start_time

                    train_steps_per_second = 1 / (time_train_step + time_to_sample)

                    train_logs = {
                        **train_logs,
                        **self.get_stats(),
                        "Environment Steps": self._env_step_ctr,
                        "Time to Sample": time_to_sample,
                        "Time for Train Step": time_train_step,
                        "Train Steps Per Second": train_steps_per_second,
                    }

                    self._logger.write(train_logs, force=True)

                if all(terminals.values()) or all(truncations.values()):
                    break

            episode_returns.append(episode_return)
            if len(episode_returns) > 40:
                episode_returns.pop(0)

            episodes += 1

            if episodes % 1 == 0:  # TODO: make variable

                env_steps_per_second = 1 / (time_to_step + time_for_action_selection)

                self._logger.write(
                    {
                        "Episodes": episodes,
                        "Episode Return": episode_return,
                        "Average Return": np.mean(episode_returns),
                        "Environment Steps": self._env_step_ctr,
                        "Time for Action Selection": time_for_action_selection,
                        "Time to Step Env": time_to_step,
                        "Time to add to Replay": time_to_add_to_replay,
                        "Env Steps Per Second": env_steps_per_second,
                    },
                    force=True,
                )

            if self._env_step_ctr > max_env_steps:
                break

    def train_offline(
        self,
        replay_buffer: FlashbaxReplayBuffer,
        max_trainer_steps: int = int(1e5),
        evaluate_every: int = 1000,
        num_eval_episodes: int = 4,
        json_writer: Optional[JsonWriter] = None,
    ) -> List:
        """Method to train the system offline.

        WARNING: make sure evaluate_every % log_every == 0 and log_every < evaluate_every,
        else you won't log evaluation.
        """
        trainer_step_ctr = 0
        while trainer_step_ctr < max_trainer_steps:
            if evaluate_every is not None and trainer_step_ctr % evaluate_every == 0:
                print("EVALUATION")
                eval_logs, render_frames = self.evaluate(num_eval_episodes)
                self._logger.write(
                    eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True
                )
                if json_writer is not None:
                    json_writer.write(
                        trainer_step_ctr,
                        "evaluator/episode_return",
                        eval_logs["evaluator/episode_return"],
                        trainer_step_ctr // evaluate_every,
                    )

            start_time = time.time()
            experience = replay_buffer.sample()
            end_time = time.time()
            time_to_sample = end_time - start_time

            start_time = time.time()
            self.counter += 1
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

            self._logger.write(logs, force=True)

            trainer_step_ctr += 1

        print("FINAL EVALUATION")
        eval_logs, render_frames = self.evaluate(10 * num_eval_episodes)
        self._logger.write(eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True)
        if json_writer is not None:
            eval_logs = {f"absolute/{key.split('/')[1]}": value for key, value in eval_logs.items()}
            json_writer.write(
                trainer_step_ctr,
                "absolute/episode_return",
                eval_logs["absolute/episode_return"],
                trainer_step_ctr // evaluate_every,
            )
            json_writer.close()

        print('\nCOUNTER: '+str(self.counter)+'\n')
        return render_frames

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
