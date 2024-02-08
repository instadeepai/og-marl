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

import copy
import time
from typing import Sequence

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
import tree
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax import linen as nn

import wandb
from og_marl.jax.dataset import FlashbaxBufferStore


def train_maicq_system(  # noqa: C901
    environment,
    logger,
    dataset_path,
    seed: int = 42,
    learning_rate: float = 3e-4,
    batch_size: float = 32,
    policy_layer_sizes: Sequence[int] = (64,),
    policy_gru_layer_size: int = 64,
    critic_layer_sizes: Sequence[int] = (64,),
    critic_gru_layer_size: int = 64,
    target_update_rate: float = 0.005,
    discount: float = 0.99,
    num_epochs: int = 100,
    num_training_steps_per_epoch: int = 1000,
    num_episodes_per_evaluation: int = 4,
    maicq_advantage_beta: float = 0.1,
    maicq_target_beta: float = 1e3,
    qmixer_embed_dim: int = 32,
    qmixer_hyper_dim: int = 64,
    json_writer=None,
):
    # GLOBAL Variables
    NUM_ACTS = environment._num_actions
    NUM_AGENTS = len(environment.possible_agents)
    SEED = seed
    LR = learning_rate
    BATCH_SIZE = batch_size
    POLICY_LAYER_SIZES = policy_layer_sizes
    POLICY_GRU_LAYER_SIZE = policy_gru_layer_size
    CRITIC_LAYER_SIZES = critic_layer_sizes
    CRITIC_GRU_LAYER_SIZE = critic_gru_layer_size
    TAU = target_update_rate
    GAMMA = discount
    NUM_EVALS = num_episodes_per_evaluation
    NUM_EPOCHS = num_epochs
    NUM_TRAIN_STEPS_PER_EPOCH = num_training_steps_per_epoch
    MAICQ_ADVANTAGE_BETA = maicq_advantage_beta
    MAICQ_TARGET_BETA = maicq_target_beta
    MIXER_EMBED_DIM = qmixer_embed_dim
    MIXER_HYPER_DIM = qmixer_hyper_dim
    DATASET_PATH = dataset_path

    # Get sequence length from dataset metadata
    store = FlashbaxBufferStore(DATASET_PATH)
    dataset_metadata = store.restore_metadata()
    SEQUENCE_LENGTH = dataset_metadata["sequence_length"]

    def stack_agents(state, agents):
        experience = {"obs": [], "act": [], "rew": [], "done": [], "legals": []}
        for agent in agents:
            experience["obs"].append(state.experience[f"{agent}_observations"])
            experience["act"].append(state.experience[f"{agent}_actions"])
            experience["rew"].append(state.experience[f"{agent}_rewards"])
            experience["done"].append(state.experience[f"{agent}_done"])
            experience["legals"].append(state.experience[f"{agent}_legals"])
        experience["obs"] = jnp.stack(experience["obs"], axis=2)
        experience["act"] = jnp.stack(experience["act"], axis=2)
        experience["rew"] = jnp.stack(experience["rew"], axis=2)
        experience["done"] = jnp.stack(experience["done"], axis=2)
        experience["legals"] = jnp.stack(experience["legals"], axis=2)
        experience["mask"] = state.experience["mask"]
        experience["env_state"] = state.experience["state"]
        state = TrajectoryBufferState(
            experience=experience, is_full=state.is_full, current_index=state.current_index
        )
        return state

    class Network(nn.Module):
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

    class QMIXER(nn.Module):
        num_agents: int
        embed_dim: int
        hyper_dim: int

        def setup(self):
            self.hyper_w_1 = nn.Sequential(
                [
                    nn.Dense(self.hyper_dim),
                    nn.relu,
                    nn.Dense(self.embed_dim * self.num_agents),
                ]
            )

            self.hyper_w_final = nn.Sequential(
                [nn.Dense(self.hyper_dim), nn.relu, nn.Dense(self.embed_dim)]
            )

            # State dependent bias for hidden layer
            self.hyper_b_1 = nn.Dense(self.embed_dim)

            # V(s) instead of a bias for the last layers
            self.V = nn.Sequential([nn.Dense(self.embed_dim), nn.relu, nn.Dense(1)])

        def __call__(self, action_values, env_state):
            B, T, N = action_values.shape[:3]

            action_values = jnp.reshape(action_values, (-1, 1, self.num_agents))
            env_state = jnp.reshape(env_state, (-1, env_state.shape[-1]))

            w1 = jnp.abs(self.hyper_w_1(env_state))
            b1 = self.hyper_b_1(env_state)

            w1 = jnp.reshape(w1, (-1, self.num_agents, self.embed_dim))
            b1 = jnp.reshape(b1, (-1, 1, self.embed_dim))

            hidden = nn.elu(jnp.matmul(action_values, w1) + b1)

            w_final = jnp.abs(self.hyper_w_final(env_state))
            w_final = jnp.reshape(w_final, (-1, self.embed_dim, 1))

            # State-dependent bias
            v = jnp.reshape(self.V(env_state), (-1, 1, 1))

            # Compute final output
            mixed_action_value = jnp.matmul(hidden, w_final) + v
            mixed_action_value = jnp.reshape(mixed_action_value, (B, -1, 1))

            return mixed_action_value

        def k(self, env_state):
            B, T, N = env_state.shape[:3]

            w1 = jnp.abs(self.hyper_w_1(env_state))
            w_final = jnp.abs(self.hyper_w_final(env_state))
            w1 = jnp.reshape(w1, (-1, self.num_agents, self.embed_dim))
            w_final = jnp.reshape(w_final, (-1, self.embed_dim, 1))
            k = jnp.matmul(w1, w_final)
            k = jnp.reshape(k, (B, -1, self.num_agents))
            k = k / (jnp.sum(k, axis=1, keepdims=True) + 1e-9)  # avoid div by zero

            return k

    def unroll_policy(params, obs_seq):
        f = lambda carry, obs: Network(POLICY_LAYER_SIZES, POLICY_GRU_LAYER_SIZE, NUM_ACTS).apply(
            params, carry, obs
        )
        init_carry = Network(POLICY_LAYER_SIZES, POLICY_GRU_LAYER_SIZE, NUM_ACTS).initialize_carry(
            POLICY_GRU_LAYER_SIZE, obs_seq.shape[1:]
        )
        carry, logits = jax.lax.scan(f, init_carry, obs_seq)
        return logits

    def unroll_critic(params, obs_seq):
        f = lambda carry, obs: Network(CRITIC_LAYER_SIZES, CRITIC_GRU_LAYER_SIZE, NUM_ACTS).apply(
            params, carry, obs
        )
        init_carry = Network(CRITIC_LAYER_SIZES, CRITIC_GRU_LAYER_SIZE, NUM_ACTS).initialize_carry(
            CRITIC_GRU_LAYER_SIZE, obs_seq.shape[1:]
        )
        carry, q_values = jax.lax.scan(f, init_carry, obs_seq)
        return q_values

    def maicq_loss(params, target_params, obs, act, rew, done, legals, env_state, mask):
        """Args:

        ----
            obs: (B,T,N,O)
            act: (B,T,N)
            rew: (B,T,N)
            done: (B,T,N)

        """
        B, T, N = obs.shape[:3]

        # Collapse agent dim into batch dim
        obs = jnp.swapaxes(obs, 1, 2)
        obs = jnp.reshape(obs, (B * N, T, obs.shape[-1]))  # (B*N,T,O)

        logits = jax.vmap(unroll_policy, (None, 0))(params["policy"], obs)
        q_values = jax.vmap(unroll_critic, (None, 0))(params["critic"], obs)
        target_q_values = jax.vmap(unroll_critic, (None, 0))(target_params["critic"], obs)

        (logits, q_values, target_q_values) = jax.tree_map(
            lambda x: jnp.swapaxes(jnp.reshape(x, (B, N, T, x.shape[2])), 1, 2),
            (logits, q_values, target_q_values),
        )

        probs = nn.softmax(logits, axis=-1)
        probs = probs * legals  # Mask illegal actions
        probs_sum = (
            jnp.sum(probs, axis=-1, keepdims=True) + 1e-9
        )  # avoid div by zero by adding small number
        probs = probs / probs_sum  # renormalise

        # Compute advantage
        action_value = jnp.sum(q_values * nn.one_hot(act, NUM_ACTS), axis=-1)
        baseline = jnp.sum(probs * q_values, axis=-1)
        advantage = action_value - baseline  # TODO: stop gradient
        advantage = nn.softmax(advantage / MAICQ_ADVANTAGE_BETA, axis=0)
        advantage = jax.lax.stop_gradient(advantage)

        action_prob = jnp.sum(probs * nn.one_hot(act, NUM_ACTS), axis=-1)
        action_logprob = jnp.log(action_prob + 1e-9)  # Added small number to avoid log of zero

        coe = QMIXER(NUM_AGENTS, MIXER_EMBED_DIM, MIXER_HYPER_DIM).apply(
            params["mixer"], env_state, method="k"
        )  # TODO check that gradients are supposed to flow here
        policy_loss = -coe * len(advantage) * advantage * action_logprob

        action_value = jnp.sum(q_values * nn.one_hot(act, NUM_ACTS), axis=-1)
        target_action_value = jnp.sum(target_q_values * nn.one_hot(act, NUM_ACTS), axis=-1)

        mixed_action_value = QMIXER(NUM_AGENTS, MIXER_EMBED_DIM, MIXER_HYPER_DIM).apply(
            params["mixer"], action_value, env_state
        )
        mixed_target_action_value = QMIXER(NUM_AGENTS, MIXER_EMBED_DIM, MIXER_HYPER_DIM).apply(
            target_params["mixer"], target_action_value, env_state
        )

        target_advantage = nn.softmax(
            mixed_target_action_value / MAICQ_TARGET_BETA,
            axis=0,  # across batch dim
        )
        target_next_value = (
            len(target_advantage) * target_advantage * mixed_target_action_value
        )  # TODO check that len is over agent dim. Or maybe time dim

        target = rew[:, :-1] + GAMMA * (1 - done[:, :-1]) * target_next_value[:, 1:]
        target = jax.lax.stop_gradient(target)

        critic_loss = 0.5 * jnp.square(target - mixed_action_value[:, :-1])

        mask = jnp.expand_dims(mask, axis=-1)

        policy_loss = jnp.sum(policy_loss * jnp.broadcast_to(mask, policy_loss.shape)) / jnp.sum(
            jnp.broadcast_to(mask, policy_loss.shape)
        )

        critic_loss = jnp.sum(
            critic_loss * jnp.broadcast_to(mask[:, :-1], critic_loss.shape)
        ) / jnp.sum(jnp.broadcast_to(mask[:, :-1], critic_loss.shape))

        loss = critic_loss + policy_loss

        return loss, {"policy_loss": policy_loss, "critic_loss": critic_loss}

    @jax.jit
    @chex.assert_max_traces(n=1)
    def train_epoch(rng_key, params, opt_state, buffer_state):
        buffer = fbx.make_trajectory_buffer(
            # NOTE: we set this to an arbitrary large number > buffer_state.current_index.
            max_length_time_axis=10_000_000,
            min_length_time_axis=BATCH_SIZE,
            sample_batch_size=BATCH_SIZE,
            add_batch_size=1,
            sample_sequence_length=SEQUENCE_LENGTH,
            period=SEQUENCE_LENGTH,
        )
        optim = optax.chain(optax.clip_by_global_norm(10), optax.adam(LR))

        def train_step(carry, rng_key):
            params, opt_state, buffer_state = carry
            batch = buffer.sample(buffer_state, rng_key)
            (loss, logs), grads = jax.value_and_grad(maicq_loss, has_aux=True)(
                params["online"],
                params["target"],
                batch.experience["obs"],
                batch.experience["act"],
                batch.experience["rew"],
                batch.experience["done"],
                batch.experience["legals"],
                batch.experience["env_state"],
                batch.experience["mask"],
            )
            updates, opt_state = optim.update(grads, opt_state, params["online"])
            params["online"] = optax.apply_updates(params["online"], updates)

            (params["target"]["critic"], params["target"]["mixer"]) = optax.incremental_update(
                (params["online"]["critic"], params["online"]["mixer"]),
                (params["target"]["critic"], params["target"]["mixer"]),
                TAU,
            )

            return (params, opt_state, buffer_state), logs

        init_carry = (params, opt_state, buffer_state)
        rng_keys = jax.random.split(rng_key, num=NUM_TRAIN_STEPS_PER_EPOCH)
        carry, logs = jax.lax.scan(train_step, init_carry, rng_keys)
        params, opt_state, buffer_state = carry
        return params, opt_state, logs

    def select_actions(carry, params, obs, legals):
        policy = Network(POLICY_LAYER_SIZES, POLICY_GRU_LAYER_SIZE, NUM_ACTS)
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
    config = {"backend": "jax"}
    wandb.init(project="benchmark-jax-og-marl", entity="claude_formanek", config=config)
    rng_key = jax.random.PRNGKey(SEED)

    # Restore Dataset
    store = FlashbaxBufferStore(DATASET_PATH)
    buffer_state = store.restore_state()
    buffer_state = stack_agents(buffer_state, environment.possible_agents)

    # Initialise Network Parameters
    dummy_obs = buffer_state.experience["obs"][0, 0]
    dumm_env_state = buffer_state.experience["env_state"][0, 0]
    policy = Network(POLICY_LAYER_SIZES, POLICY_GRU_LAYER_SIZE, NUM_ACTS)
    init_policy_carry = policy.initialize_carry(POLICY_GRU_LAYER_SIZE, dummy_obs.shape)
    policy_params = policy.init(rng_key, init_policy_carry, dummy_obs)
    critic = Network(CRITIC_LAYER_SIZES, CRITIC_GRU_LAYER_SIZE, NUM_ACTS)
    init_critic_carry = critic.initialize_carry(CRITIC_GRU_LAYER_SIZE, dummy_obs.shape)
    critic_params = critic.init(rng_key, init_critic_carry, dummy_obs)
    dummy_multi_agent_qvals = jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH, NUM_AGENTS))
    dummy_multi_dim_env_state = jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH, dumm_env_state.shape[0]))
    mixer = QMIXER(NUM_AGENTS, MIXER_EMBED_DIM, MIXER_HYPER_DIM)
    mixer_params = mixer.init(rng_key, dummy_multi_agent_qvals, dummy_multi_dim_env_state)

    params = {
        "online": {"policy": policy_params, "critic": critic_params, "mixer": mixer_params},
        "target": {"critic": copy.deepcopy(critic_params), "mixer": copy.deepcopy(mixer_params)},
    }

    opt_state = optax.chain(optax.clip_by_global_norm(10), optax.adam(LR)).init(params["online"])

    for i in range(NUM_EPOCHS):
        eval_logs = evaluation(init_policy_carry, params["online"]["policy"], environment)
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

        logs["critic_loss"] = jnp.mean(logs["critic_loss"])
        logs["policy_loss"] = jnp.mean(logs["policy_loss"])
        logs["Trainer Steps"] = (i + 1) * NUM_TRAIN_STEPS_PER_EPOCH
        if i != 0:  # don't log SPC when tracing
            logs["Train SPS"] = 1 / ((end_time - start_time) / NUM_TRAIN_STEPS_PER_EPOCH)

    eval_logs = evaluation(init_policy_carry, params["online"]["policy"], environment)
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
