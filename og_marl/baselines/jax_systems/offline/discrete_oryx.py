import copy
from functools import partial
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import hydra
import jax
import jax.numpy as jnp
from jax import tree
import numpy as np
import optax
from flashbax.vault import Vault
from flashbax.buffers.flat_buffer import TrajectoryBuffer, TrajectoryBufferState
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint
from typing_extensions import NamedTuple

from og_marl.environments import get_environment
from og_marl.vault_utils.download_vault import download_and_unzip_vault

from og_marl.baselines.jax_systems.networks import OryxNetwork
from og_marl.baselines.jax_systems.types import ExperimentOutput, LearnerFn, Metrics
from og_marl.baselines.jax_systems.utils.logger import LogEvent, MavaLogger
from og_marl.baselines.jax_systems.networks.utils.oryx import get_init_hidden_state
from og_marl.baselines.jax_systems.utils.network_utils import get_action_head
from og_marl.baselines.jax_systems.utils.training import make_learning_rate
from og_marl.baselines.jax_systems.types import Observation
from og_marl.baselines.jax_systems.systems.oryx.types import (
    ActorApply,
    LearnerApply,
    Params,
    Transition,
)

_DISCRETE = "discrete"

class LearnerState(NamedTuple):
    """State of the offline learner for Memory Sable"""

    params: Params
    opt_states: OptState
    key: Any
    buffer_state: TrajectoryBufferState
    steps: int

def map_og_marl_to_transition(env_info, buffer_state):
    old_exp_full = buffer_state.experience
    old_exp = old_exp_full

    # Shit dones by one timestep
    terminals = jnp.concatenate((jnp.zeros((1,1,env_info["num_agents"]), dtype=old_exp_full['terminals'].dtype), old_exp_full['terminals'][:,:-1,...]), axis=1)

    # Compute step counts for each episode
    def _cumsum(current_step,done):
        next_step = (current_step+1)*(1-done)
        return next_step, next_step
    
    def _scan_cumsum(seq):
        init_carry = -1.0  # Initial carry value.
        _, cumulative_sums = jax.lax.scan(_cumsum, init_carry, seq)
        return cumulative_sums

    full_step_count = jax.vmap(_scan_cumsum)(terminals[:,:,0])
    new_step_count = jnp.tile(full_step_count[:,:,jnp.newaxis],(1,1,terminals.shape[2]))


    # Add agent IDs to Obs
    obs = old_exp['observations']
    agent_ids = jnp.tile(jnp.eye(env_info["num_agents"], dtype=obs.dtype)[jnp.newaxis, jnp.newaxis, ...], (*obs.shape[:2],1,1))
    obs = jnp.concat((agent_ids, obs), axis=-1)

    # Build the sample into a RecQTransition
    new_experience = Transition(
        done = terminals,
        action = old_exp['actions'],
        reward = old_exp['rewards'],
        obs = Observation(
            agents_view = obs,
            action_mask = old_exp['infos']['legals'],
            step_count = new_step_count, 
        ),
        info = {
            'episode_length' : jnp.zeros_like(old_exp['terminals'][:,:,jnp.newaxis,...]),
            'episode_return' : jnp.zeros_like(old_exp['terminals'][:,:,jnp.newaxis,...]),
            'is_terminal_step' : terminals[:,:,jnp.newaxis,...],
        }
    )
    return TrajectoryBufferState(experience=new_experience,current_index=jnp.array(0, dtype='int32'),is_full=jnp.array(True, dtype=bool))


def evaluate(env, params, key, net_config, select_actions_fn, num_eval_episodes: int = 32):
        """Method to evaluate the system in an OG-MARL wrapped environment."""
        episode_returns = []
        try:
            prev_wins = env._environment.get_stats()["battles_won"]
        except:
            prev_wins = 0

        for _ in range(num_eval_episodes):
            observations, infos = env.reset()

            done = False
            episode_return = 0.0
            step_count = 0
            hstate = get_init_hidden_state(net_config, 1)
            while not done:

                legal_actions = infos["legals"]

                key, policy_key = jax.random.split(key, 2)

                stacked_obs = jnp.stack(list(observations.values()), axis=0)[jnp.newaxis,...]
                agent_ids = jnp.expand_dims(jnp.eye(env.num_agents, dtype=stacked_obs.dtype), axis=0)
                stacked_obs = jnp.concat((agent_ids, stacked_obs), axis=-1)

                actions, _, _, hstate = select_actions_fn(
                    params,
                    Observation(
                        agents_view=stacked_obs,
                        action_mask=jnp.stack(list(legal_actions.values()), axis=0)[jnp.newaxis,...],
                        step_count=jnp.zeros((1,env.num_agents)) + step_count,
                    ),
                    hstate,
                    policy_key,
                )

                act_dict = {}
                for i in range(actions.shape[1]):
                    act_dict[f"agent_{i}"] = int(actions[0,i])
                
                observations, rewards, terminal, truncation, infos = env.step(act_dict)

                episode_return += np.mean(list(rewards.values()), dtype="float")

                done = all(terminal.values()) or all(truncation.values())

                step_count += 1

            episode_returns.append(episode_return)

        try:
            wins = (env._environment.get_stats()["battles_won"] -  prev_wins) / num_eval_episodes
        except:
            wins = 0

        logs = {
            "win_rate": wins,
            "mean_episode_return": np.mean(episode_returns),
            "max_episode_return": np.max(episode_returns),
            "min_episode_return": np.min(episode_returns),
        }

        return logs


def get_learner_fn(
    apply_fns: Tuple[ActorApply, LearnerApply],
    update_fn: optax.TransformUpdateFn,
    buffer: TrajectoryBuffer,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply functions for executing and training the network.
    action_select, apply_fn = apply_fns

    def _update_step(learner_state: LearnerState) -> Tuple[LearnerState, Tuple[Transition, Metrics]]:
        """A single update of the network."""

        # UNPACK NEW LEARNER STATE
        (
            params,
            opt_states,
            key,
            buffer_state,
            steps,
        ) = learner_state

        def _update(train_state: Tuple, batch: Transition) -> Tuple:
            """Update the network for one gradient step."""
            # UNPACK TRAIN STATE
            params, opt_state, key = train_state

            def _loss_fn(
                online_params: FrozenDict,
                target_params: FrozenDict,
                batch: Transition,
            ) -> Tuple:
                """Calculate Oryx loss."""
                B = batch.action.shape[0]
                N = config.system.num_agents

                concat_time_and_agents = lambda x: jnp.reshape(
                    x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])
                )

                batch = jax.tree.map(concat_time_and_agents, batch)

                if config.system.env_name == "rware":
                    # The shape in the rware dataset is (B, N, 1), which is not like the other datasets
                    batch = Transition(
                        done = batch.done,
                        action = batch.action.squeeze(-1),
                        reward = batch.reward,
                        obs = batch.obs,
                        info = batch.info,
                    )

                # Get the predicted logits and q-values.
                logits, q_values = apply_fn(  # type: ignore
                    online_params,
                    observation=batch.obs,
                    action=batch.action,
                    dones=batch.done,
                    hstates=get_init_hidden_state(
                        config.network.net_config, B
                    ),
                )

                # Select predicted q values of the actions from the replay buffer.
                action_q_value = jnp.squeeze(
                    jnp.take_along_axis(
                        q_values, 
                        jnp.expand_dims(batch.action, axis=-1).astype(jnp.int32), 
                        axis=-1
                    ), 
                    axis=-1
                )

                # Get the target logits and q-values.
                target_logits, target_q_values = apply_fn(  # type: ignore
                    target_params,
                    observation=batch.obs,
                    action=batch.action,
                    dones=batch.done,
                    hstates=get_init_hidden_state(
                        config.network.net_config, B
                    ),
                )

                # ICQ: Select target q values according to batch actions
                target_next_action_value = jnp.squeeze(
                    jnp.take_along_axis(
                        target_q_values,
                        jnp.expand_dims(batch.action, axis=-1).astype(jnp.int32),
                        axis=-1,
                    ),
                    axis=-1,
                )
                advantage_q = jax.nn.softmax(target_next_action_value / config.system.value_temperature, axis=0) # across batch dim
                target_next_action_value = len(advantage_q) * advantage_q * target_next_action_value

                # Compute target
                target = batch.reward[:,:-N] + (
                    config.system.gamma * (1 - batch.done[:,N:]) * target_next_action_value[:,N:]
                )
                target = jax.lax.stop_gradient(target)

                # TD Error
                td_error = 0.5 * jnp.square(target - action_q_value[:,:-N]).mean()

                # Policy Loss
                policy_probs = jax.nn.softmax(logits, axis=-1)
                baseline = jnp.sum(policy_probs * q_values, axis=-1) # Coma

                # Compute advantage (stop gradient to prevent affecting Q-network)
                advantage = jax.lax.stop_gradient(action_q_value - baseline)

                # ICQ: Compute softmax advantage for weighting
                advantage_weight = jax.lax.stop_gradient(
                    jax.nn.softmax(advantage / config.system.policy_temperature, axis=0)
                ) * len(advantage)

                # Extract policy probability for taken action
                action_prob = jnp.take_along_axis(
                    policy_probs,
                    jnp.expand_dims(batch.action, axis=-1).astype(jnp.int32),
                    axis=-1,
                )

                # Compute advantage weighted policy gradient loss
                policy_loss = jnp.mean(-1 * advantage_weight * jnp.log(action_prob).squeeze(-1))
 
                # Compute total loss by combining critic and policy loss
                loss =  config.system.critic_coef * td_error + policy_loss

                return loss, (td_error, policy_loss, action_prob.mean(), action_prob.min(), action_prob.max(), action_prob.std(), action_q_value.mean(),
                                advantage_weight.mean(), advantage_weight.min(), advantage_weight.max(), advantage_weight.std(),
                                advantage_q.mean(), advantage_q.min(), advantage_q.max(), advantage_q.std()
                            )

            # CALCULATE LOSS
            key, training_key = jax.random.split(key)
            
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            loss_info, grads = grad_fn(params.online, params.target, batch)

            # UPDATE PARAMS AND OPTIMISER STATE
            updates, new_opt_state = update_fn(grads, opt_state)
            online_new_params = optax.apply_updates(params.online, updates)

            # Target network polyak update.
            target_new_params = optax.incremental_update(
                online_new_params, params.target, config.system.tau
            )

            new_params = Params(online_new_params, target_new_params)

            # PACK LOSS INFO
            loss_info = {
                "loss": loss_info[0],
                "td_error": loss_info[1][0],
                "policy_loss": loss_info[1][1],
                "action_prob_mean": loss_info[1][2],
                "action_prob_min": loss_info[1][3],
                "action_prob_max": loss_info[1][4],
                "action_prob_std": loss_info[1][5],
                "action_q_value": loss_info[1][6],
                "advantage_weight_mean": loss_info[1][7],
                "advantage_weight_min": loss_info[1][8],
                "advantage_weight_max": loss_info[1][9],
                "advantage_weight_std": loss_info[1][10],
                "advantage_q_mean": loss_info[1][11],
                "advantage_q_min": loss_info[1][12], 
                "advantage_q_max": loss_info[1][13],
                "advantage_q_std": loss_info[1][14],
            }

            return (new_params, new_opt_state, key), loss_info

        # Sample a batch of trajectories
        key, buffer_sample_key, agent_shuffle_key, training_key = jax.random.split(key, 4)
        batch = buffer.sample(buffer_state, buffer_sample_key).experience

        # Shuffle agents
        agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
        batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=2), batch)  # NBNB

        # Update network according to the batch
        (params, opt_states, _), loss_info = _update((params, opt_states, training_key), batch)

        learner_state = LearnerState(
            params=params,
            opt_states=opt_states,
            key=key,
            buffer_state=buffer_state,
            steps=steps+1,
        )

        return learner_state, loss_info

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function."""

        def update_fn(carry, _):
            """Update function for the scan."""
            new_carry, info = _update_step(carry)
            return new_carry, info

        # Use jax.lax.scan to iterate the update step.
        learner_state, loss_info = jax.lax.scan(update_fn, learner_state, None, length=int(config.system.num_updates_per_eval))


        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics={},
            train_metrics=loss_info,
        )

    return learner_fn

def learner_setup(
    env_info: Dict, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Callable, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available devices.
    n_devices = len(jax.devices())

    # PRNG keys.
    key, net_key = keys

    # Get number of agents.
    config.system.num_agents = env_info["num_agents"]
    n_agents = env_info["num_agents"]

    # Get number of agents and actions.
    action_dim = env_info["num_actions"]
    config.system.num_actions = action_dim

    # Setting the chunksize
    config.network.memory_config.chunk_size = config.system.sample_sequence_length * n_agents

    _, action_space_type = get_action_head(_DISCRETE)

    # Define network.
    sable_q_network = OryxNetwork(
        n_agents=n_agents,
        n_agents_per_chunk=n_agents,
        action_dim=action_dim,
        net_config=config.network.net_config,
        memory_config=config.network.memory_config,
        action_space_type=action_space_type,
    )

    # Define optimiser.
    lr = make_learning_rate(config.system.lr, config)
    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-8),
    )

    # Get mock inputs to initialise network.
    init_obs =Observation(
            agents_view=jnp.zeros((1,env_info["num_agents"],env_info["obs_dim"]+env_info["num_agents"])),
            action_mask=jnp.zeros((1,env_info["num_agents"],env_info["num_actions"])),
            step_count=jnp.zeros((1,env_info["num_agents"])),
    )
    init_hs = get_init_hidden_state(config.network.net_config, 1)
    init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # Initialise params and optimiser state.
    online_params = sable_q_network.init(
        net_key,
        init_obs,
        init_hs,
        net_key,
        method="get_actions",
    )

    opt_state = optim.init(online_params)

    params = Params(online_params, copy.deepcopy(online_params))

    # Pack apply and update functions.
    # Using dummy hstates, since we are not updating the hstates during training.
    apply_fns = (
        partial(sable_q_network.apply, method="get_actions"),  # Execution function
        partial(sable_q_network.apply),  # Training function
    )
    eval_apply_fn = partial(sable_q_network.apply, method="get_actions")

    # Initialise trajectory buffer
    rb = fbx.make_trajectory_buffer(
        sample_sequence_length=config.system.sample_sequence_length,
        period=1, 
        add_batch_size=1, # dummy since we never add data
        sample_batch_size=config.system.sample_batch_size,
        max_length_time_axis=424242, # dummy
        min_length_time_axis=424242, # dummy
    )

    # Download and load OG-MARL dataset
    download_and_unzip_vault(env_info["source"], env_info["env_name"], env_info['scenario_name'])
    vlt = Vault(f"{env_info['source']}/{env_info['env_name']}/{env_info['scenario_name']}.vlt", vault_uid=env_info['dataset'])
    buffer_state_vlt = vlt.read(percentiles=(0,100))

    # Map OG-MARL dataset to RecQTransition
    buffer_state = map_og_marl_to_transition(env_info, buffer_state_vlt)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(apply_fns, optim.update, rb, config)
    learn = jax.jit(learn)

    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)
    replicate_learner = (params, opt_state, step_keys, jnp.zeros(1))

    # Initialise learner state.
    params, opt_state, step_keys, n_env_steps = replicate_learner

    init_learner_state = LearnerState(
        params=params,
        opt_states=opt_state,
        key=step_keys,
        buffer_state=buffer_state,
        steps=n_env_steps,
    )


    return learn, eval_apply_fn, init_learner_state, vlt


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "offline_rec_q_sable"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Initialise environment from OG-MARL
    eval_env = get_environment(config["system"]["source"], config["system"]["env_name"], config["system"]["scenario"], config["system"]["seed"])

    # Get shapes
    dummy_env = get_environment(config["system"]["source"], config["system"]["env_name"], config["system"]["scenario"], config["system"]["seed"])
    dummy_obs, info = dummy_env.reset()
    obs_dim = dummy_obs["agent_0"].shape[-1]
    # dummy_env._environment.close()
    del dummy_env

    # Get env info
    env_info = {
        "num_agents": eval_env.num_agents,
        "num_actions": eval_env.num_actions,
        "scenario_name": config["system"]["scenario"],
        "env_name": config["system"]["env_name"],
        "dataset":  config["system"]["dataset"],
        "source":  config["system"]["source"],
        "obs_dim": obs_dim
    }
    # PRNG keys.
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=3)

    # Setup learner.
    learn, execution_fn, learner_state, _ = learner_setup(env_info, (key, net_key), config)
    execution_fn = jax.jit(execution_fn)

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.system.num_evaluation

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    pprint(cfg)

    # First Evaluation
    trained_params = learner_state.params.online
    eval_logs = evaluate(eval_env, trained_params, key_e, config.network.net_config, execution_fn, config.system.num_eval_episodes)
    eval_step = 0
    logger.log(eval_logs, eval_step, eval_step, LogEvent.EVAL)

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.system.num_evaluation):
        # Train.
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        logger.log(learner_output.train_metrics, (eval_step+1)*config.system.num_updates_per_eval, (eval_step+1)*config.system.num_updates_per_eval, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = learner_output.learner_state.params.online
        key, key_e = jax.random.split(key, 2)

        # Evaluate.
        eval_logs = evaluate(eval_env, trained_params, key_e, config.network.net_config, execution_fn, config.system.num_eval_episodes)
        logger.log(eval_logs, (eval_step+1)*config.system.num_updates_per_eval, (eval_step+1)*config.system.num_updates_per_eval, LogEvent.EVAL)
        episode_return = jnp.mean(eval_logs["mean_episode_return"])

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

        # Absolute metric as per MARL-eval
        if max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

    # Measure absolute metric.
    eval_logs = evaluate(eval_env, best_params, key_e, config.network.net_config, execution_fn, config.system.num_eval_episodes * 10)
    eval_step+=2
    logger.log(eval_logs, eval_step*config.system.num_updates_per_eval, eval_step*config.system.num_updates_per_eval, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return episode_return


@hydra.main(
    config_path="configs/default",
    config_name="oryx.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Set task name
    cfg.system.task_name = cfg.system.source + "_" + cfg.system.env_name+ "_" + cfg.system.scenario + "_" + cfg.system.dataset

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"Offline Oryx experiment completed")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
