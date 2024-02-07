# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import time

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence

import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
import tree
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax import linen as nn

import wandb
from og_marl.jax.tf_dataset_to_flashbax import FlashbaxBufferStore


def train_bc_system(  # noqa: C901
    environment,
    logger,
    dataset_path,
    seed: int = 42,
    learning_rate: float = 3e-4,
    batch_size: float = 32,
    policy_layer_sizes: Sequence[int] = (64,),
    policy_gru_layer_size: int = 64,
    num_epochs: int = 1000,
    num_training_steps_per_epoch: int = 1000,
    num_episodes_per_evaluation: int = 4,
    json_writer=None,
):
    ##################
    ##### Config #####
    ##################

    BATCH_SIZE = batch_size
    LR = learning_rate
    LAYER_SIZES = policy_layer_sizes
    GRU_LAYER_SIZE = policy_gru_layer_size
    NUM_TRAIN_STEPS_PER_EPOCH = num_training_steps_per_epoch
    NUM_EVALS = num_episodes_per_evaluation
    DATASET_PATH = dataset_path
    SEED = seed
    NUM_EPOCHS = num_epochs

    NUM_ACTS = environment._num_actions
    # NUM_AGENTS = len(environment.possible_agents)

    ##################
    ### End Config ###
    ##################

    def stack_agents(state, agents):
        experience = {"obs": [], "act": []}
        for agent in agents:
            experience["obs"].append(state.experience[f"{agent}_observations"])
            experience["act"].append(state.experience[f"{agent}_actions"])
        experience["obs"] = jnp.stack(experience["obs"], axis=2)
        experience["act"] = jnp.stack(experience["act"], axis=2)
        experience["mask"] = state.experience["mask"]
        state = TrajectoryBufferState(
            experience=experience, is_full=state.is_full, current_index=state.current_index
        )
        return state

    class BehaviourCloningPolicy(nn.Module):
        dense_layer_sizes: Sequence[int]
        gru_hidden_size: int
        output_size: int

        @nn.compact
        def __call__(self, carry, inputs):
            dense_layers = [nn.Dense(size) for size in self.dense_layer_sizes]
            x = inputs
            for layer in dense_layers:
                x = layer(x)
                x = nn.relu(x)
            carry, x = nn.GRUCell(self.gru_hidden_size)(carry, x)

            # Final dense layer for output
            output = nn.Dense(self.output_size)(x)

            return carry, output

        @staticmethod
        def initialize_carry(layer_size, input_shape):
            """Initializes the carry state."""
            # Use a dummy key since the default state init fn is just zeros.
            return nn.GRUCell(layer_size).initialize_carry(jax.random.PRNGKey(0), input_shape)

    def softmax_cross_entropy_loss(index, logits):  # softmax cross entropy loss
        num_labels = logits.shape[-1]
        labels = nn.one_hot(index, num_labels)
        probs = nn.softmax(logits)
        return -jnp.sum(
            labels * jnp.log(probs + 1e-12), axis=-1
        )  # small constant for numerical stability

    def unroll_policy(params, obs_seq):
        f = lambda carry, obs: BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS).apply(
            params, carry, obs
        )
        init_carry = BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS).initialize_carry(
            GRU_LAYER_SIZE, obs_seq.shape[-1:]
        )
        carry, logits = jax.lax.scan(f, init_carry, obs_seq)
        return logits

    def behaviour_cloning_loss(params, obs_seq, act_seq, mask):
        logits = unroll_policy(params, obs_seq)
        logits = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            logits,
            jnp.ones_like(logits),
        )  # avoid nans, get masked out later
        loss = jax.vmap(softmax_cross_entropy_loss)(act_seq, logits)
        return jnp.sum(loss * mask) / jnp.sum(mask)  # masked mean

    def batched_multi_agent_behaviour_cloninig_loss(params, obs_seq, act_seq, mask):
        """Args:

        ----
            params: a container of params for the behaviour cloning network which is shared between
                all agents in the system.
            obs_seq: an array of a sequence of observations for all agents. Shape (B,N,T,O) where
                B is the batch dim, N is the number of agents, T is the time dimension and O is
                the observation dim.
            act_seq: is an array of a sequence of actions for all agents. Shape (B,N,T).

        Returns:
        -------
            A scalar behaviour cloning loss.

        """
        multi_agent_behaviour_cloning_loss = jax.vmap(
            behaviour_cloning_loss, (None, 1, 1, None)
        )  # vmap over agent dim which is after time dim
        batched_multi_agent_behaviour_cloninig_loss = jax.vmap(
            multi_agent_behaviour_cloning_loss, (None, 0, 0, 0)
        )
        loss = batched_multi_agent_behaviour_cloninig_loss(params, obs_seq, act_seq, mask)
        return jnp.mean(loss)

    def train_epoch(rng_key, params, opt_state, buffer_state):
        buffer = fbx.make_trajectory_buffer(
            # NOTE: we set this to an arbitrary large number > buffer_state.current_index.
            max_length_time_axis=10_000_000,
            min_length_time_axis=BATCH_SIZE,
            sample_batch_size=BATCH_SIZE,
            add_batch_size=1,
            sample_sequence_length=20,
            period=20,
        )
        optim = optax.chain(optax.clip_by_global_norm(10), optax.adam(LR))

        def train_step(carry, rng_key):
            params, opt_state, buffer_state = carry
            batch = buffer.sample(buffer_state, rng_key)
            loss, grads = jax.value_and_grad(batched_multi_agent_behaviour_cloninig_loss)(
                params, batch.experience["obs"], batch.experience["act"], batch.experience["mask"]
            )
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, buffer_state), loss

        init_carry = (params, opt_state, buffer_state)
        rng_keys = jax.random.split(rng_key, num=NUM_TRAIN_STEPS_PER_EPOCH)
        carry, loss = jax.lax.scan(train_step, init_carry, rng_keys)
        params, opt_state, buffer_state = carry
        return params, opt_state, {"loss": loss}

    def select_actions(carry, params, obs, legals):
        policy = BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS)
        carry, logits = policy.apply(params, carry, obs)
        logits = jnp.where(legals, logits, -99999999.0)  # Legal action masking
        act = jnp.argmax(logits, axis=-1)
        return carry, act

    def stack(obs, agents):
        stacked_obs = []
        for agent in agents:
            stacked_obs.append(obs[agent])
        stacked_obs = tree.map_structure(lambda x: jnp.array(x), stacked_obs)
        stacked_obs = jnp.stack(stacked_obs)
        return stacked_obs

    def unstack(stacked_values, agents):
        unstacked = {}
        for i, agent in enumerate(agents):
            unstacked[agent] = int(stacked_values[i])
        return unstacked

    def evaluation(init_carry, params, environment):
        episode_returns = []
        for _ in range(NUM_EVALS):
            episode_return = 0
            done = False
            obs, info = environment.reset()
            carry = init_carry
            while not done:
                obs = stack(obs, environment.possible_agents)
                legals = stack(info["legals"], environment.possible_agents)
                carry, act = jax.jit(select_actions)(carry, params, obs, legals)
                act = unstack(act, environment.possible_agents)

                obs, rew, term, trunc, info = environment.step(act)

                done = all(trunc.values()) or all(term.values())
                episode_return += sum(list(rew.values())) / len(
                    list(rew.values())
                )  # mean over agents
            episode_returns.append(episode_return)
        return {"evaluator/episode_return": sum(episode_returns) / NUM_EVALS}

    ################
    ##### MAIN #####
    ################

    wandb.init(project="jax-og-marl")
    rng_key = jax.random.PRNGKey(SEED)
    store = FlashbaxBufferStore(DATASET_PATH)
    buffer_state = store.restore_state()
    buffer_state = stack_agents(buffer_state, environment.possible_agents)
    dummy_obs = buffer_state.experience["obs"][0, 0, 0]
    policy = BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS)
    init_carry = policy.initialize_carry(GRU_LAYER_SIZE, dummy_obs.shape)
    params = policy.init(rng_key, init_carry, dummy_obs)
    opt_state = optax.chain(optax.clip_by_global_norm(10), optax.adam(3e-4)).init(params)

    for i in range(NUM_EPOCHS):
        eval_logs = evaluation(init_carry, params, environment)
        logger.write(eval_logs, force=True)
        if json_writer is not None:
            json_writer.write(
                (i + 1) * NUM_TRAIN_STEPS_PER_EPOCH,
                "evaluator/episode_return",
                eval_logs["evaluator/episode_return"],
                i,
            )

        start_time = time.time()
        rng_key, train_key = jax.random.split(rng_key)
        params, opt_state, logs = train_epoch(train_key, params, opt_state, buffer_state)
        end_time = time.time()

        logs["loss"] = jnp.mean(logs["loss"])
        logs["Trainer Steps"] = (i + 1) * NUM_TRAIN_STEPS_PER_EPOCH
        if i != 0:  # don't log SPC when tracing
            logs["Train SPS"] = 1 / ((end_time - start_time) / NUM_TRAIN_STEPS_PER_EPOCH)

    eval_logs = evaluation(init_carry, params, environment)
    logger.write(eval_logs, force=True)
    if json_writer is not None:
        eval_logs = {f"absolute/{key.split('/')[1]}": value for key, value in eval_logs.items()}
        json_writer.write(
            (i + 1) * NUM_TRAIN_STEPS_PER_EPOCH,
            "absolute/episode_return",
            eval_logs["absolute/episode_return"],
            i,
        )

    print("Done")
