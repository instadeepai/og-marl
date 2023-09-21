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

from typing import Dict
import tensorflow as tf
import sonnet as snt

from og_marl.systems.trainer_base import TrainerBase
from og_marl.utils.trainer_utils import (
    sample_batch_agents,
    batch_concat_agent_id_to_obs, 
    expand_batch_and_agent_dim_of_time_major_sequence, 
    merge_batch_and_agent_dim_of_time_major_sequence, 
    switch_two_leading_dims
)


class DiscreteBCTrainer(TrainerBase):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        behaviour_cloning_network: snt.Module,
        optimizer: snt.Optimizer,
        max_gradient_norm: float = 20.0,
        add_agent_id_to_obs: bool = False
    ):
        """Initialise discrete action trainer."""

        super().__init__(
            agents,
            dataset,
            logger,
            discount=None,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs
        )

        self._optimizer = optimizer

        # Behaviour cloning network
        self._behaviour_cloning_network = behaviour_cloning_network

        # Expose the network variables.
        self._system_variables: Dict = {
            "behaviour_cloning_network": behaviour_cloning_network.variables,
        }

    @tf.function
    def _train(self, sample, trainer_step):

        # Batch agent inputs together
        batch = sample_batch_agents(self._agents, sample)

        # Get max sequence length, batch size, num agents and num actions
        B, T, N, A = batch["legals"].shape

        # Unpack the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        mask = batch["mask"]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)

        # Merge batch_dim and agent_dim
        observations = merge_batch_and_agent_dim_of_time_major_sequence(observations)

        with tf.GradientTape(persistent=True) as tape:

            # Compute policy
            probs_out, _ = snt.static_unroll(
                self._behaviour_cloning_network, 
                observations, 
                self._behaviour_cloning_network.initial_state(B*N)
            )

             # Expand batch and agent_dim
            probs_out = expand_batch_and_agent_dim_of_time_major_sequence(probs_out, B, N)

            # Make batch-major again
            probs_out = switch_two_leading_dims(probs_out)

            # Behaviour cloning loss
            one_hot_actions = tf.one_hot(actions, depth=probs_out.shape[-1], axis=-1)
            bc_mask = tf.concat([mask] * N, axis=-1)
            probs_out = tf.where(
                tf.cast(tf.expand_dims(bc_mask, axis=-1), "bool"),
                probs_out,
                1 / A * tf.ones(A, "float32"),
            )  # avoid nans, get masked out later
            bc_loss = tf.keras.metrics.categorical_crossentropy(
                one_hot_actions, probs_out
            )
            bc_loss = tf.reduce_sum(bc_loss * bc_mask) / tf.reduce_sum(bc_mask)

        # Apply gradients to policy
        variables = (
            *self._behaviour_cloning_network.trainable_variables,
        )  # Get trainable variables

        gradients = tape.gradient(bc_loss, variables)  # Compute gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[
            0
        ]  # Maybe clip gradients.

        self._optimizer.apply(gradients, variables)

        del tape

        return {"BC Loss": bc_loss, "Trainer Steps": trainer_step}

class ContinuousBCTrainer(DiscreteBCTrainer):
    def __init__(
        self,
        agents,
        dataset,
        logger,
        behaviour_cloning_network,
        optimizer,
        max_gradient_norm=20.0,
        add_agent_id_to_obs=False,
    ):
        super().__init__(
            agents=agents,
            dataset=dataset,
            optimizer=optimizer,
            behaviour_cloning_network=behaviour_cloning_network,
            logger=logger,
            max_gradient_norm=max_gradient_norm,
            add_agent_id_to_obs=add_agent_id_to_obs,
        )

    @tf.function
    def _train(self, sample, trainer_step):
        batch = sample_batch_agents(self._agents, sample)

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        mask = tf.cast(batch["mask"], "float32")  # shape=(B,T)

        # Get dims
        B, T, N = legal_actions.shape[:3]

        # Maybe add agent ids to observation
        if self._add_agent_id_to_obs:
            observations = batch_concat_agent_id_to_obs(observations)

        # Make time-major
        observations = switch_two_leading_dims(observations)
        actions = switch_two_leading_dims(actions)
        mask = switch_two_leading_dims(mask)

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape() as tape:
            # Unroll network
            a_out, _ = snt.static_unroll(
                self._behaviour_cloning_network,
                merge_batch_and_agent_dim_of_time_major_sequence(observations),
                self._behaviour_cloning_network.initial_state(B*N)
            )
            a_out = expand_batch_and_agent_dim_of_time_major_sequence(a_out, B, N)

            # BC loss
            bc_loss = (a_out - actions) ** 2

            # Masking zero-padded elements
            mask = tf.concat([mask] * N, axis=-1)
            bc_loss = tf.reduce_sum(tf.expand_dims(mask, axis=-1) * bc_loss) / tf.reduce_sum(mask)

        # Get trainable variables
        variables = (*self._behaviour_cloning_network.trainable_variables,)

        # Compute gradients.
        gradients = tape.gradient(bc_loss, variables)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        self._optimizer.apply(gradients, variables)

        del tape

        logs = {
            "Trainer Steps": trainer_step,
            "BC Loss": bc_loss,
        }

        return logs
