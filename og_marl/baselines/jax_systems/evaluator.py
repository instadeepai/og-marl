# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import warnings
from typing import Any, Callable, Dict, Protocol, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from og_marl.baselines.jax_systems.env_types import MarlEnv

from og_marl.baselines.jax_systems.types import (
    Action,
    ActorApply,
    Metrics,
    Observation,
    ObservationGlobalState,
    RecActorApply,
    State,
)
from og_marl.baselines.jax_systems.wrappers.gym import GymToJumanji

# Optional extras that are passed out of the actor and then into the actor in the next step
ActorState: TypeAlias = Dict[str, Any]
# Type of the carry for the _env_step function in the evaluator
_EvalEnvStepState: TypeAlias = Tuple[State, TimeStep, PRNGKey, ActorState]
# The function signature for the mava evaluation function (returned by `get_eval_fn`).
EvalFn: TypeAlias = Callable[[FrozenDict, PRNGKey, ActorState], Metrics]


class EvalActFn(Protocol):
    """The API for the acting function that is passed to the `EvalFn`.

    A get_action function must conform to this API in order to be used with Mava's evaluator.
    See `make_ff_eval_act_fn` and `make_rec_eval_act_fn` as examples.
    """

    def __call__(
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        actor_state: ActorState,
    ) -> Tuple[Array, ActorState]: ...


def get_num_eval_envs(config: DictConfig, absolute_metric: bool) -> int:
    """Returns the number of vmapped envs/batch size during evaluation."""
    n_devices = jax.device_count()
    n_parallel_envs = 16

    if absolute_metric:
        eval_episodes = config.system.num_eval_episodes * 10
    else:
        eval_episodes = config.system.num_eval_episodes

    return math.ceil(eval_episodes / n_devices)  # type: ignore


def get_eval_fn(
    env: MarlEnv, act_fn: EvalActFn, config: DictConfig, absolute_metric: bool
) -> EvalFn:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
    ----
        env: an environment that conforms to the mava environment spec.
        act_fn: a function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: the system config.
        absolute_metric: whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    eval_episodes = (
        config.system.num_eval_episodes * 10
        if absolute_metric
        else config.system.num_eval_episodes
    )
    n_vmapped_envs = get_num_eval_envs(config, absolute_metric)
    n_parallel_envs = n_vmapped_envs
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.

        Metrics are collected by the `RecordEpisodeMetrics` wrapper: episode return and length,
        also win rate for environments that support it.

        Returns: Dict[str, Array] - dictionary of metric name to metric values for each episode.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, n_vmapped_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"]
            if config.env.log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # find the first instance of done to get the metrics at that timestep, we don't
            # care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, jnp.arange(n_vmapped_envs)], metrics)
            del metrics["is_terminal_step"]  # uneeded for logging

            return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes.
        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        metrics = eval_fn(params, key, init_act_state)
        metrics = jax.block_until_ready(metrics)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn