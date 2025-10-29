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

from typing import Callable, Tuple, Union, Any

import chex
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict as Params
from omegaconf import DictConfig
from og_marl.baselines.jax_systems.systems.oryx.types import Transition
from og_marl.baselines.jax_systems.systems.oryx.types import (
    ActorApply,
    LearnerApply,
)
from og_marl.baselines.jax_systems.types import Observation
from og_marl.baselines.jax_systems.networks.utils.oryx import get_init_hidden_state

def make_learning_rate_schedule(init_lr: float, config: DictConfig) -> Callable:
    """Makes a very simple linear learning rate scheduler.

    Args:
    ----
        init_lr: initial learning rate.
        config: system configuration.

    Note:
    ----
        We use a simple linear learning rate scheduler based on the suggestions from a blog on PPO
        implementation details which can be viewed at http://tinyurl.com/mr3chs4p
        This function can be extended to have more complex learning rate schedules by adding any
        relevant arguments to the system config and then parsing them accordingly here.

    """

    def linear_scedule(count: int) -> float:
        frac: float = (
            1.0
            - (count // (config.system.ppo_epochs * config.system.num_minibatches))
            / config.system.num_updates
        )
        return init_lr * frac

    return linear_scedule


def make_learning_rate(init_lr: float, config: DictConfig) -> Union[float, Callable]:
    """Retuns a constant learning rate or a learning rate schedule.

    Args:
    ----
        init_lr: initial learning rate.
        config: system configuration.

    Returns:
    -------
        A learning rate schedule or fixed learning rate.

    """
    return init_lr

def compute_flops(
    params: Params,
    apply_fns: Tuple[ActorApply, LearnerApply],
    init_data: Tuple[Observation, Array],
    config: DictConfig,
    net_key: PRNGKey,
) -> DictConfig:
    init_obs, init_hs = init_data

    # Compute inference flops for a single step
    # We don't do inference flops for offline systems. 

    base_train_shape = (1, config.system.sample_sequence_length, config.system.num_agents)

    _done = jnp.zeros(base_train_shape, dtype=bool)
    _action = jnp.ones(base_train_shape, dtype=int)
    _reward = jnp.ones(base_train_shape, dtype=float)
    _done_mask = jnp.ones(base_train_shape, dtype=bool)
    _obs = jax.tree.map(lambda x: x[:, None, ...].repeat(config.system.sample_sequence_length, axis=1), init_obs)

    _batch = Transition(
        done=_done,
        action=_action,
        reward=_reward,
        done_mask=_done_mask,
        obs=_obs,
        train_mask=None,
    )
    
    def _loss_fn(
        params: Params,
        batch: Transition,
    ) -> Tuple:
        """Calculate Sable loss."""
        B = batch.action.shape[0]

        batch_t = jax.tree.map(lambda x: x[:, :-1, ...], batch)
        batch_next_t = jax.tree.map(lambda x: x[:, 1:, ...], batch)

        concat_time_and_agents = lambda x: jnp.reshape(
            x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])
        )

        batch_t = jax.tree.map(concat_time_and_agents, batch_t)
        batch_next_t = jax.tree.map(concat_time_and_agents, batch_next_t)

        # Get the predicted q values.
        logits, q_values = apply_fns[1](  # type: ignore
            params,
            observation=batch_t.obs,
            action=batch_t.action,
            dones=batch_t.done,
            hstates=get_init_hidden_state(
                config.network.net_config, B
            ),
        )

        # Select predicted q values of the actions from the replay buffer.
        action_q_value = jnp.squeeze(
            jnp.take_along_axis(
                q_values, 
                jnp.expand_dims(batch_t.action, axis=-1).astype(jnp.int32), 
                axis=-1
            ), 
            axis=-1
        )
        
        _, target_q_values = apply_fns[1](  # type: ignore
            params,
            observation=batch_next_t.obs,
            action=batch_next_t.action,
            dones=batch_next_t.done,
            hstates=get_init_hidden_state(
                config.network.net_config, B
            ),
        )

        # Select target q values according to online predicted actions.
        target_next_action_value = jnp.squeeze(
            jnp.take_along_axis(
                target_q_values,
                jnp.expand_dims(batch_next_t.action, axis=-1).astype(jnp.int32),
                axis=-1,
            ),
            axis=-1,
        )

        advantage_q = jax.nn.softmax(target_next_action_value / config.system.softmax_temperature_target, axis=0) # across batch dim
        target_next_action_value = advantage_q.shape[0] * advantage_q * target_next_action_value
        if config.system.use_mask_done:
            target = batch_t.reward + (
                config.system.gamma * (1 - batch_next_t.done_mask) * target_next_action_value
            )
        else:
            target = batch_t.reward + (
                config.system.gamma * (1 - batch_next_t.done) * target_next_action_value
            )
        # TD Error
        td_error = 0.5 * jnp.square(target - action_q_value).mean() #where=jnp.logical_and(batch_t.train_mask, batch_next_t.train_mask))

        # Policy Loss
        policy_probs = jax.nn.softmax(logits, axis=-1)
        state_value = jnp.sum(policy_probs * q_values, axis=-1)
        advantage = jax.lax.stop_gradient(action_q_value - state_value)

        advantage_softmax = jax.nn.softmax(advantage / config.system.softmax_temperature_coma, axis=0)
        action_prob = jnp.take_along_axis(
                policy_probs,
                jnp.expand_dims(batch_t.action, axis=-1).astype(jnp.int32),
                axis=-1,
            )
        # jax.debug.print("{x}", x=action_log_prob.min())
        #TODO: not sure if we need the train mask here
        coma_loss = -jnp.mean((len(advantage_softmax) * advantage_softmax * jnp.log(action_prob).squeeze(-1)))
        # coma_loss = action_log_prob.mean()

        loss =  td_error + coma_loss


        return loss

    # Compute forward pass flops for a single training minibatch
    grad_fn = jax.grad(_loss_fn)
    jit_grad_fn = jax.jit(grad_fn)
    compiled = jit_grad_fn.lower(
        params,
        _batch,
    ).compile()

    train_flops = compiled.cost_analysis()[0]["flops"]    

    # Set FLOPs data in config
    if "compute_cost" not in config:
        config.compute_cost = {}
    config.compute_cost.train_flops = (
        train_flops
        * config.system.epochs
        * config.system.update_batch_size
        * config.system.sample_batch_size # because the loss_fn uses data with a batch size of 1
    )
    return config


def count_params(params: Params, config: DictConfig) -> DictConfig:
    # Flatten the parameter tree and sum up the sizes of all parameter arrays.
    param_leaves = jax.tree.leaves(params)
    total_params = sum(param.size for param in param_leaves if isinstance(param, jnp.ndarray))

    if "compute_cost" not in config:
        config.compute_cost = {}
    config.compute_cost.param_count = total_params
    return config