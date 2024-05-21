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
from absl import app, flags
import copy
import time

import jax
import flashbax as fbx
from flashbax.buffers import sum_tree
from flashbax.vault import Vault
import tensorflow as tf
import sonnet as snt
import numpy as np
import matplotlib.pyplot as plt

from og_marl.environments import get_environment
from og_marl.loggers import WandbLogger
from og_marl.offline_dataset import download_and_unzip_vault
from og_marl.tf2.utils import set_growing_gpu_memory

set_growing_gpu_memory()

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "mamujoco", "Environment name.")
flags.DEFINE_string("scenario", "2halfcheetah", "Environment scenario name.")
flags.DEFINE_string("dataset", "Good", "Dataset type.")
flags.DEFINE_string("system", "maddpg+bc", "System name.")
flags.DEFINE_string("joint_action", "buffer", "")
flags.DEFINE_float("trainer_steps", 3e5, "Number of training steps.")
flags.DEFINE_float("priority_exponent", 0.99, "Priority exponent")
flags.DEFINE_float("gaussian_steepness", 4., "")
flags.DEFINE_float("bc_alpha", 2.5, "")
flags.DEFINE_integer("prioritised_batch_size", 256, "")
flags.DEFINE_integer("uniform_batch_size", 100000, "")
flags.DEFINE_integer("update_priorities_every", 10, "")
flags.DEFINE_integer("seed", 42, "Seed.")


class TransitionBuffer:
    def __init__(
        self,
        env_name,
        scenario_name,
        dataset_name,
        rel_dir: str = "vaults",
        prioritised_batch_size=256,
        uniform_batch_size=1024,
        priority_exponent=0.99,
        seed=42,
    ):
        self._rng_key = jax.random.PRNGKey(seed)

        # Load data from Vault
        self._vault_buffer_state = Vault(
            vault_name=f"{env_name}/{scenario_name}.vlt",
            vault_uid=dataset_name,
            rel_dir=rel_dir,
        ).read()

        # Get example timestep
        example_timestep = jax.tree_map(lambda x: x[0, 0], self._vault_buffer_state.experience)

        # Initialise uniform buffer
        self._uniform_replay_buffer = fbx.make_prioritised_flat_buffer(
            max_length=self._vault_buffer_state.experience["truncations"].shape[1] + 1,
            min_length=1,
            add_batch_size=1,
            add_sequences=True,
            sample_batch_size=uniform_batch_size,
            priority_exponent=0.0,  # No prioritisation
            device="gpu",
        )

        # Initialise prioritised buffer
        self._prioritised_replay_buffer = fbx.make_prioritised_flat_buffer(
            max_length=self._vault_buffer_state.experience["truncations"].shape[1] + 1,
            min_length=1,
            add_batch_size=1,
            add_sequences=True,
            sample_batch_size=prioritised_batch_size,
            priority_exponent=priority_exponent,
            device="gpu",
        )
        tmp_buffer_state = self._prioritised_replay_buffer.init(example_timestep)

        self._uniform_sample_fn = jax.jit(self._uniform_replay_buffer.sample)
        self._prioritised_sample_fn = jax.jit(self._prioritised_replay_buffer.sample)
        self._set_priorities_fn = jax.jit(
            self._prioritised_replay_buffer.set_priorities,
            donate_argnums=0,
        )

        self._buffer_state = self._prioritised_replay_buffer.add(
            tmp_buffer_state, self._vault_buffer_state.experience
        )

    def uniform_sample(self):
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._uniform_sample_fn(self._buffer_state, sample_key)
        return batch

    def prioritised_sample(self):
        self._rng_key, sample_key = jax.random.split(self._rng_key, 2)
        batch = self._prioritised_sample_fn(self._buffer_state, sample_key)
        return batch

    def update_priorities(self, indices, priorities):
        self._buffer_state = self._set_priorities_fn(
            self._buffer_state, indices, priorities.numpy()
        )


class StateAndJointActionCritic(snt.Module):
    def __init__(self, num_agents: int, num_actions: int):
        self.N = num_agents
        self.A = num_actions

        self._critic_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(1),
            ]
        )

        super().__init__()

    def __call__(
        self,
        states,
        agent_actions,
        other_actions,
    ):
        """Forward pass of critic network.

        observations [B,N,O]
        states [B,S]
        agent_actions [B,N,A]: the actions the agent took.
        other_actions [B,N,A]: the actions the other agents took.
        """
        # Make joint action
        joint_actions = self.make_joint_action(agent_actions, other_actions)

        # Repeat states for each agent
        states = tf.stack([states] * self.N, axis=1)  # [B,S] -> [B,N,S]

        # Concat states and joint actions
        critic_input = tf.concat([states, joint_actions], axis=-1)

        q_values = self._critic_network(critic_input)

        return q_values

    def make_joint_action(self, agent_actions, other_actions):
        """Method to construct the joint action.

        agent_actions [B,N,A]: tensor of actions the agent took. Usually
            the actions from the learnt policy network.
        other_actions [[B,N,A]]: tensor of actions the agent took. Usually
            the actions from the replay buffer.
        """
        B, N, A = agent_actions.shape[:3]  # (B,N,A)
        all_joint_actions = []
        for i in range(N):  # type: ignore
            one_hot = tf.expand_dims(
                tf.cast(tf.stack(tf.stack([tf.one_hot(i, N)] * B, axis=0), axis=0), "bool"),  # type: ignore
                axis=-1,
            )
            joint_action = tf.where(one_hot, agent_actions, other_actions)
            joint_action = tf.reshape(joint_action, (B, N * A))  # type: ignore
            all_joint_actions.append(joint_action)
        all_joint_actions = tf.stack(all_joint_actions, axis=1)

        return all_joint_actions


class FFMADDPG:
    def __init__(
        self,
        env,
        buffer,
        logger,
        target_update_rate=0.005,
        critic_learning_rate=3e-4,
        policy_learning_rate=3e-4,
        bc_alpha=2.5,
        update_priorities_every=None,
        bc_reg=False,
        gaussian_steepness=1,
    ):
        self.env = env
        self.buffer = buffer
        self.logger = logger

        # Policy network
        self.policy_network = snt.Sequential(
            [
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(self.env._num_actions),
                tf.nn.tanh,
            ]
        )  # shared network for all agents

        # Target policy network
        self.target_policy_network = copy.deepcopy(self.policy_network)

        # Critic network
        self.critic_network_1 = StateAndJointActionCritic(
            len(self.env.possible_agents), self.env._num_actions
        )  # shared network for all agents
        self.critic_network_2 = copy.deepcopy(self.critic_network_1)

        # Target critic network
        self.target_critic_network_1 = copy.deepcopy(self.critic_network_1)
        self.target_critic_network_2 = copy.deepcopy(self.critic_network_1)
        self.target_update_rate = target_update_rate

        # Optimizers
        self.critic_optimizer = snt.optimizers.Adam(learning_rate=critic_learning_rate)
        self.policy_optimizer = snt.optimizers.Adam(learning_rate=policy_learning_rate)

        # Offline Regularisers
        self.bc_reg = bc_reg
        self.discount = 0.99
        self.update_priorities_every = update_priorities_every
        self.priority_on_ramp = 150_000
        self.gaussian_steepness = gaussian_steepness
        self.bc_alpha = bc_alpha

    @tf.function(jit_compile=True)
    def select_actions(self, observations):
        actions = {}
        agents = list(observations.keys())
        for i, agent in enumerate(agents):
            agent_observation = observations[agent]

            agent_observation = concat_agent_id_to_obs(agent_observation, i, len(agents))

            agent_observation = tf.expand_dims(agent_observation, axis=0)  # add batch dimension
            actions[agent] = self.policy_network(agent_observation)[0]  # unbatch

        return actions

    @tf.function(jit_compile=True)
    def compute_new_priorities(self, experience, trainer_step):
        observations = experience.first["observations"]  # (B,N,O)
        actions = experience.first["actions"]  # (B,N,A)

        N, A = actions.shape[-2:]

        observations = batch_concat_agent_id_to_obs(observations)

        target_actions = self.target_policy_network(observations)
        noise = tf.clip_by_value(tf.random.normal(target_actions.shape, 0, 0.2), -0.5, 0.5)
        target_actions = target_actions + noise
        target_actions = tf.clip_by_value(target_actions, -1, 1)

        distance = tf.reduce_mean(
            tf.reduce_mean(tf.abs(actions - target_actions), axis=-1), axis=-1
        )  # L1

        priority_on_ramp = tf.minimum(1.0, trainer_step * (1/self.priority_on_ramp))
        priority = tf.exp(-((self.gaussian_steepness * priority_on_ramp * distance) ** 2))

        priority = tf.clip_by_value(priority, 0.001, 1.)

        logs = {
            "Max Priority": tf.reduce_max(priority),
            "Mean Priority": tf.reduce_mean(priority),
            "Min Priority": tf.reduce_min(priority),
            "STD Priority": tf.math.reduce_std(priority),
            "Mean action distance": tf.reduce_mean(distance),
            "Max action distance": tf.reduce_max(distance),
            "Min action distance": tf.reduce_min(distance),
            "STD action distance": tf.math.reduce_std(distance),
            "priority on ramp": priority_on_ramp
        }

        return logs, priority

    @tf.function(jit_compile=True)
    def train_step(self, experience, train_step):
        # Unpack the batch
        observations = experience.first["observations"]  # (B,N,O)
        next_observations = experience.second["observations"]  # (B,N,O)
        actions = experience.first["actions"]  # (B,N,A)
        env_states = experience.first["infos"]["state"]  # (B,S)
        next_env_states = experience.second["infos"]["state"]  # (B,S)
        rewards = experience.first["rewards"]  # (B,N)
        terminals = tf.cast(experience.first["terminals"], "float32")  # (B,N)

        # Get dims
        B, N, A = actions.shape[:3]
        O = observations.shape[2]

        # Add agent ids to observation
        observations = batch_concat_agent_id_to_obs(observations)
        next_observations = batch_concat_agent_id_to_obs(next_observations)

        # Target policy
        determ_target_actions = self.target_policy_network(next_observations)
        noise = tf.clip_by_value(tf.random.normal(determ_target_actions.shape, 0, 0.2), -0.5, 0.5)
        target_actions = determ_target_actions + noise
        target_actions = tf.clip_by_value(target_actions, -1, 1)

        # Target critics
        target_qs_1 = self.target_critic_network_1(next_env_states, target_actions, target_actions)
        target_qs_2 = self.target_critic_network_2(next_env_states, target_actions, target_actions)

        # Take minimum between two target critics
        target_qs = tf.minimum(target_qs_1, target_qs_2)
        target_qs = tf.squeeze(target_qs, axis=-1)

        # Compute Bellman targets
        targets = rewards + self.discount * (1 - terminals) * target_qs

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            ###############
            # Policy Loss #
            ###############
            
            # Online policy
            online_actions = self.policy_network(observations)

            # Evaluate action
            policy_qs_1 = self.critic_network_1(env_states, online_actions, actions)
            policy_qs_2 = self.critic_network_2(env_states, online_actions, actions)
            policy_qs = tf.minimum(policy_qs_1, policy_qs_2)

            if self.bc_reg:
                ##########
                # BC Reg #
                ##########
                bc_lambda = self.bc_alpha / tf.reduce_mean(tf.abs(tf.stop_gradient(policy_qs)))
                policy_loss = tf.reduce_mean((actions - online_actions) ** 2) - bc_lambda * tf.reduce_mean(policy_qs)  # + 1e-3 * tf.reduce_mean(online_actions**2)

            ###############
            # Critic Loss #
            ###############

            # Online critics
            qs_1 = tf.squeeze(
                self.critic_network_1(env_states, actions, actions),
                axis=-1,
            )
            qs_2 = tf.squeeze(
                self.critic_network_2(env_states, actions, actions),
                axis=-1,
            )

            # Squared TD-error
            critic_loss_1 = tf.reduce_mean(0.5 * (targets - qs_1) ** 2)
            critic_loss_2 = tf.reduce_mean(0.5 * (targets - qs_2) ** 2)

            # Combine critic losses
            critic_loss = (critic_loss_1 + critic_loss_2) / 2.0

        # Update critics
        variables = (
            *self.critic_network_1.trainable_variables,
            *self.critic_network_2.trainable_variables,
        )
        gradients = tape.gradient(critic_loss, variables)
        self.critic_optimizer.apply(gradients, variables)

        # Update policy
        variables = (*self.policy_network.trainable_variables,)
        gradients = tape.gradient(policy_loss, variables)
        self.policy_optimizer.apply(gradients, variables)

        # Update target networks
        online_variables = (
            *self.critic_network_1.variables,
            *self.critic_network_2.variables,
            *self.policy_network.variables,
        )
        target_variables = (
            *self.target_critic_network_1.variables,
            *self.target_critic_network_2.variables,
            *self.target_policy_network.variables,
        )
        # Soft target update
        for src, dest in zip(online_variables, target_variables):
            dest.assign(dest * (1.0 - self.target_update_rate) + src * self.target_update_rate)

        del tape

        distance = tf.reduce_mean(tf.reduce_mean(tf.abs(online_actions - actions), axis=-1), axis=-1)

        logs = {
            "Mean Q-values": tf.reduce_mean((qs_1 + qs_2) / 2),
            "Mean Critic Loss": critic_loss,
            "Mean Sample Distance": tf.reduce_mean(distance),
            "Min Sample Distance": tf.reduce_min(distance),
            "Max Sample Distance": tf.reduce_max(distance),
            "STD Sample Distance": tf.math.reduce_std(distance),
        }

        return logs


def concat_agent_id_to_obs(obs, agent_id, N):
    agent_id = tf.one_hot(agent_id, depth=N)

    obs = tf.concat([agent_id, obs], axis=-1)

    return obs


def batch_concat_agent_id_to_obs(obs):
    B, N = obs.shape[:2]  # batch size, num_agents

    agent_ids = []
    for i in range(N):  # type: ignore
        agent_id = tf.one_hot(i, depth=N)
        agent_ids.append(agent_id)
    agent_ids = tf.stack(agent_ids, axis=0)

    # Repeat along batch dim
    agent_ids = tf.stack([agent_ids] * B, axis=0)

    obs = tf.concat([agent_ids, obs], axis=-1)

    return obs


def evaluate(env, system, num_eval_episodes = 4):
    """Method to evaluate the system online (i.e. in the environment)."""
    episode_returns = []
    for _ in range(num_eval_episodes):
        # Reset
        observations, _ = env.reset()
        done = False
        episode_return = 0.0
        while not done:
            actions = system.select_actions(observations)

            observations, rewards, terminals, truncations, infos = env.step(actions)
            episode_return += np.mean(list(rewards.values()), dtype="float")
            done = all(terminals.values()) or all(truncations.values())
        episode_returns.append(episode_return)
    logs = {"evaluator/episode_return": np.mean(episode_returns)}
    return logs


def train_offline(
    env, system, buffer, logger, max_trainer_steps=1e6, evaluate_every=5000, num_eval_episodes=4
):
    trainer_step_ctr = 0
    while trainer_step_ctr < max_trainer_steps:
        if evaluate_every is not None and trainer_step_ctr % evaluate_every == 0:
            print("EVALUATION")
            eval_logs = evaluate(env, system, num_eval_episodes)
            logger.write(eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True)

        start_time = time.time()
        data_batch = buffer.prioritised_sample()
        end_time = time.time()
        time_to_sample = end_time - start_time

        # Log sampled Priorities
        indices = data_batch.indices
        old_priorities = sum_tree.get(buffer._buffer_state.priority_state, indices)

        start_time = time.time()
        train_logs = system.train_step(data_batch.experience, tf.convert_to_tensor(trainer_step_ctr))
        end_time = time.time()
        time_train_step = end_time - start_time

        if (
            system.update_priorities_every is not None
            and trainer_step_ctr % system.update_priorities_every == 0
            and trainer_step_ctr >= 1000
        ):

            # Plot Priorities  
            # if trainer_step_ctr % 500 == 0:
            #     indices = jax.numpy.arange(buffer._buffer_state.current_index)
            #     old_priorities = sum_tree.get(buffer._buffer_state.priority_state, indices)

            #     plt.bar(indices[-256:], old_priorities[-256:])
            #     plt.savefig("priorities.png")
            #     plt.clf()

            start_time = time.time()
            distance_batch = buffer.uniform_sample()
            distance_logs, new_priorities = system.compute_new_priorities(distance_batch.experience, tf.convert_to_tensor(trainer_step_ctr, "float32"))
            buffer.update_priorities(distance_batch.indices, new_priorities)
            end_time = time.time()
            time_priority = end_time - start_time
            distance_logs["Priority Update Time"] = time_priority
        else:
            distance_logs = {}

        train_steps_per_second = 1 / (time_train_step + time_to_sample)

        logs = {
            **train_logs,
            **distance_logs,
            "Trainer Steps": trainer_step_ctr,
            "Time to Sample": time_to_sample,
            "Time for Train Step": time_train_step,
            "Train Steps Per Second": train_steps_per_second,
            "Mean Sampled Priorities": np.mean(old_priorities),
            "Max Sampled Priorities": np.max(old_priorities),
            "Min Sampled Priorities": np.min(old_priorities),
            "STD Sampled Priorities": np.std(old_priorities),
        }

        logger.write(logs)

        trainer_step_ctr += 1

    print("FINAL EVALUATION")
    eval_logs = evaluate(env, system, num_eval_episodes)
    logger.write(eval_logs | {"Trainer Steps (eval)": trainer_step_ctr}, force=True)


def main(_):
    env = get_environment(FLAGS.env, FLAGS.scenario)

    download_and_unzip_vault(FLAGS.env, FLAGS.scenario)

    buffer = TransitionBuffer(
        FLAGS.env,
        FLAGS.scenario,
        FLAGS.dataset,
        prioritised_batch_size=FLAGS.prioritised_batch_size,
        uniform_batch_size=FLAGS.uniform_batch_size,
        seed=FLAGS.seed,
    )

    logger = WandbLogger(entity="off-the-grid-marl-team", project="ff-maddpg")

    system_kwargs = {
        "bc_reg": True,
        "update_priorities_every": FLAGS.update_priorities_every
        if FLAGS.system == "maddpg+bc+per" else None,
        "gaussian_steepness": FLAGS.gaussian_steepness,
        "bc_alpha": FLAGS.bc_alpha,
    }

    system = FFMADDPG(env, buffer, logger, **system_kwargs)

    train_offline(
        env, system, buffer, logger, max_trainer_steps=FLAGS.trainer_steps, evaluate_every=5000
    )


if __name__ == "__main__":
    app.run(main)
