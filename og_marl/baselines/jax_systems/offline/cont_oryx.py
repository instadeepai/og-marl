import copy
from functools import partial
from typing import Any, Callable, Dict, Tuple, NamedTuple
from gymnasium.spaces import Discrete, Box
import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flashbax.vault import Vault
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from flashbax.buffers.flat_buffer import TrajectoryBuffer, TrajectoryBufferState
from og_marl.baselines.jax_systems.networks.cont_oryx_network import ContOryxNetwork
from og_marl.baselines.jax_systems.networks.utils.cont_oryx import get_init_hidden_state
from og_marl.baselines.jax_systems.systems.oryx.types import (
    ActorApply,
    LearnerApply,
    Params,
    Transition,
    OptState,
)
from og_marl.baselines.jax_systems.types import Observation
from og_marl.baselines.jax_systems.types import ExperimentOutput, LearnerFn, Metrics
from og_marl.baselines.jax_systems.utils.checkpointing import Checkpointer
from og_marl.baselines.jax_systems.utils.logger import LogEvent, MavaLogger
from og_marl.baselines.jax_systems.utils.network_utils import get_action_head
from og_marl.baselines.jax_systems.utils.training import make_learning_rate
import tensorflow_probability.substrates.jax.distributions as tfd
from og_marl.vault_utils.download_vault import download_and_unzip_vault
from og_marl.environments import get_environment

_CONTINUOUS = "continuous"

from jax import config

# config.update('jax_disable_jit', True)
config.update("jax_debug_nans", True)

class LearnerState(NamedTuple):
    """State of the offline learner for Memory Sable"""

    params: Params
    opt_states: OptState
    key: Any
    buffer_state: TrajectoryBufferState
    steps: int

def map_og_marl_to_transition(env_info, buffer_state):
    old_exp_full = buffer_state.experience

    # Each step takes prev_done
    terminals = jnp.concatenate((jnp.zeros((1,1,env_info["num_agents"]), dtype=old_exp_full['terminals'].dtype), old_exp_full['terminals'][:,:-1,...]), axis=1)

    def _cumsum(current_step,done):
        next_step = (current_step+1)*(1-done)
        return next_step, next_step
    
    # chatgpt helped
    def _scan_cumsum(seq):
        init_carry = -1.0  # Initial carry value.
        _, cumulative_sums = jax.lax.scan(_cumsum, init_carry, seq)
        return cumulative_sums

    full_step_count = jax.vmap(_scan_cumsum)(terminals[:,:,0])
    new_step_count = jnp.tile(full_step_count[:,:,jnp.newaxis],(1,1,terminals.shape[2]))

    obs = old_exp_full['observations']
    agent_ids = jnp.tile(jnp.eye(env_info["num_agents"], dtype=obs.dtype)[jnp.newaxis, jnp.newaxis, ...], (*obs.shape[:2],1,1))
    obs = jnp.concat((agent_ids, obs), axis=-1)

    # build the sample into a Transition
    new_experience = Transition(
        done = terminals,
        action = old_exp_full['actions'],
        reward = old_exp_full['rewards'],
        obs = Observation(
            agents_view = obs,
            action_mask = None,
            step_count = new_step_count, 
        ),
    )
    return TrajectoryBufferState(experience=new_experience,current_index=jnp.array(0, dtype='int32'),is_full=jnp.array(True, dtype=bool))

def evaluate(env, params, key, net_config, select_actions_fn, num_eval_episodes: int = 32):
    """Method to evaluate the system in the environment."""
    episode_returns = []

    for _ in range(num_eval_episodes):
        observations, infos = env.reset()
        done = False
        episode_return = 0.0
        step_count = 0
        # Initialize hidden state at the start of each episode
        hstate = get_init_hidden_state(net_config, 1)
        
        while not done:
            # legal_actions = infos["legals"]

            key, policy_key = jax.random.split(key, 2)

            stacked_obs = jnp.stack(list(observations.values()), axis=0)[jnp.newaxis,...]
            agent_ids = jnp.expand_dims(jnp.eye(len(env.agents), dtype=stacked_obs.dtype), axis=0)
            stacked_obs = jnp.concat((agent_ids, stacked_obs), axis=-1)

            # Pass and update hidden state
            actions, _, hstate = select_actions_fn(
                params,
                Observation(
                    agents_view=stacked_obs,
                    action_mask=None,
                    step_count=jnp.zeros((1,len(env.agents))) + step_count
                ),
                hstate,  # Pass hidden state
                policy_key,
            )

            act_dict = {}
            for i in range(actions.shape[1]):
                act_dict[f"agent_{i}"] = actions[0,i]
            
            observations, rewards, terminal, truncation, infos = env.step(act_dict)

            episode_return += np.mean(list(rewards.values()), dtype="float")

            done = all(terminal.values()) or all(truncation.values())

            step_count += 1

        episode_returns.append(episode_return)

    logs = {
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
    sable_action_select, sable_apply_fn = apply_fns

    def _update_step(learner_state: LearnerState) -> Tuple[LearnerState, Tuple[Transition, Metrics]]:
        """A single update of the network.
        This function does no environment interactions. It only updates the learner.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The current model parameters.
                - opt_states (OptState): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.

        """

        # UNPACK NEW LEARNER STATE
        (
            params,
            opt_states,
            key,
            buffer_state,
            steps,
        ) = learner_state

        def _update(train_state: Tuple, batch: Transition) -> Tuple:
            """Update the network for a single epoch."""
            # UNPACK TRAIN STATE
            params, opt_state, key = train_state

            def _loss_fn(
                online_params: FrozenDict,
                target_params: FrozenDict,
                batch: Transition,
            ) -> Tuple:
                """Calculate Sable loss."""
                B = batch.action.shape[0]
                N = config.system.num_agents

                concat_time_and_agents = lambda x: jnp.reshape(
                    x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])
                )

                batch = jax.tree.map(concat_time_and_agents, batch)

                # Get the predicted q values.
                online_action, log_probs, q_values, entropy  = sable_apply_fn(  # type: ignore
                    online_params,
                    observation=batch.obs,
                    action=batch.action,
                    dones=batch.done,
                    hstates=get_init_hidden_state(
                        config.network.net_config, B
                    ),
                    rng_key=key,
                )

                action_q_value = q_values #+ value.squeeze(-1)
                
                _, _, target_next_q_values, _ = sable_apply_fn(  # type: ignore
                    target_params,
                    observation=batch.obs,
                    action=batch.action,
                    dones=batch.done,
                    hstates=get_init_hidden_state(
                        config.network.net_config, B
                    ),
                    rng_key=key,
                )

                # ICQ: Select target q values according to batch actions
                target_next_action_value = target_next_q_values #+ target_value.squeeze(-1)

                advantage_q = jax.nn.softmax(target_next_action_value / config.system.value_temperature, axis=0) # across batch dim
                target_next_action_value = advantage_q.shape[0] * advantage_q * target_next_action_value

                target = batch.reward[:,:-N] + (
                    config.system.gamma * (1 - batch.done[:,N:]) * target_next_action_value[:,N:].squeeze(-1)
                )

                # TD Error
                td_error = 0.5 * jnp.square(target - action_q_value[:,:-N].squeeze(-1)).mean()

                v = target_next_q_values
                advantage = q_values.squeeze(-1) - v.squeeze(-1)
                advantage = jax.lax.stop_gradient(advantage)
                advantage_softmax = jax.nn.softmax(advantage / config.system.policy_temperature, axis=0)

                # jax.debug.print("{x}", x=action_log_prob.min())
                policy_loss = -jnp.mean((advantage_softmax.shape[0] * advantage_softmax * log_probs))

                loss = policy_loss + config.system.critic_coef * td_error

                # Calculate metrics for logging
                advantage_weight_mean = advantage_softmax.mean()
                advantage_weight_min = advantage_softmax.min()
                advantage_weight_max = advantage_softmax.max()
                advantage_weight_std = advantage_softmax.std()
                
                advantage_q_mean = advantage_q.mean()
                advantage_q_min = advantage_q.min()
                advantage_q_max = advantage_q.max()
                advantage_q_std = advantage_q.std()

                return loss, (td_error, policy_loss, q_values.mean(), q_values.min(), q_values.max(), q_values.std(), 
                            target_next_action_value.mean(), target_next_action_value.min(), target_next_action_value.max(), target_next_action_value.std(),
                            advantage_weight_mean, advantage_weight_min, advantage_weight_max, advantage_weight_std,
                            advantage_q_mean, advantage_q_min, advantage_q_max, advantage_q_std, entropy)

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

            loss_info = {
                "loss": loss_info[0],
                "td_error": loss_info[1][0],
                "policy_loss": loss_info[1][1],
                "action_q_value_mean": loss_info[1][2],
                "action_q_value_min": loss_info[1][3],
                "action_q_value_max": loss_info[1][4],
                "action_q_value_std": loss_info[1][5],
                "target_next_action_value_mean": loss_info[1][6],
                "target_next_action_value_min": loss_info[1][7],
                "target_next_action_value_max": loss_info[1][8],
                "target_next_action_value_std": loss_info[1][9],
                "advantage_weight_mean": loss_info[1][10],
                "advantage_weight_min": loss_info[1][11],
                "advantage_weight_max": loss_info[1][12],
                "advantage_weight_std": loss_info[1][13],
                "advantage_q_mean": loss_info[1][14],
                "advantage_q_min": loss_info[1][15],
                "advantage_q_max": loss_info[1][16],
                "advantage_q_std": loss_info[1][17],
                "entropy": loss_info[1][18]
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
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The initial model parameters.
                - opt_state (OptState): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.

        """

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

    # PRNG keys.
    key, net_key = keys

    # Get number of agents and actions.
    action_dim = env_info["num_actions"]
    n_agents = env_info["num_agents"]
    config.system.num_agents = n_agents
    config.system.num_actions = action_dim

    # Setting the chunksize for recurrent network
    config.network.memory_config.chunk_size = (config.system.sample_sequence_length) * n_agents

    # Set positional encoding to True for recurrent network
    config.network.memory_config.timestep_positional_encoding = True

    _, action_space_type = get_action_head(_CONTINUOUS)

    # Define network.
    sable_sac_network = ContOryxNetwork(
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
    init_obs = Observation(
            agents_view=jnp.zeros((1,env_info["num_agents"],env_info["obs_dim"]+env_info["num_agents"])),
            action_mask=None,
            step_count=jnp.zeros((1,env_info["num_agents"]))
    )
    init_actions = jnp.zeros((1,env_info["num_agents"],env_info["num_actions"]))
    init_hs = get_init_hidden_state(config.network.net_config, 1)

    # Initialise params and optimiser state.
    online_params = sable_sac_network.init(
        net_key,
        init_obs,
        # init_actions,
        init_hs,
        # dones=jnp.zeros((1,env_info["num_agents"])),
        key=net_key,
        method="get_actions",
    )
    opt_state = optim.init(online_params)

    params = Params(online_params, copy.deepcopy(online_params))

    # Pack apply and update functions for recurrent network
    apply_fns = (
        partial(
            sable_sac_network.apply, method="get_actions"
        ),  # Execution function
        partial(sable_sac_network.apply),  # Training function
    )
    eval_apply_fn = partial(sable_sac_network.apply, method="get_actions")

    # Initialise trajectory buffer
    rb = fbx.make_trajectory_buffer(
        # n transitions gives n-1 full data points
        sample_sequence_length=config.system.sample_sequence_length, # for recurrent network
        period=1,  # sample any unique trajectory
        add_batch_size=1,
        sample_batch_size=config.system.sample_batch_size,
        max_length_time_axis=42, # dummy
        min_length_time_axis=42, # dummy
    )

    download_and_unzip_vault(env_info["source"], env_info["env_name"], env_info['scenario_name'])
    vlt = Vault(f"{env_info['source']}/{env_info['env_name']}/{env_info['scenario_name']}.vlt", vault_uid=env_info['dataset'])
    buffer_state_vlt = vlt.read(percentiles=(0,100))

    buffer_state = map_og_marl_to_transition(env_info, buffer_state_vlt)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(apply_fns, optim.update, rb, config)
    learn = jax.jit(learn)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

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
    _config.logger.system_name = "offline_cont_oryx"
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    eval_env = get_environment(config["system"]["source"], config["system"]["env_name"], config["system"]["scenario"], config["system"]["seed"])
    
    obs_dim = list(eval_env.reset()[0].values())[0].shape[0]

    env_info = {
        "num_agents": len(eval_env.agents),
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
    learn, sable_execution_fn, learner_state, _ = learner_setup(env_info, (key, net_key), config)
    sable_execution_fn = jax.jit(sable_execution_fn)

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.system.num_evaluation

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    pprint(cfg)

    # First Evaluation
    trained_params = learner_state.params.online
    eval_logs = evaluate(eval_env, trained_params, key_e, config.network.net_config, sable_execution_fn, config.system.num_eval_episodes)
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
        eval_logs = evaluate(eval_env, trained_params, key_e, config.network.net_config, sable_execution_fn, config.system.num_eval_episodes)
        logger.log(eval_logs, (eval_step+1)*config.system.num_updates_per_eval, (eval_step+1)*config.system.num_updates_per_eval, LogEvent.EVAL)
        episode_return = jnp.mean(eval_logs["mean_episode_return"])

        if max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    eval_logs = evaluate(eval_env, best_params, key_e, config.network.net_config, sable_execution_fn, config.system.num_eval_episodes * 10)
    eval_step+=2
    logger.log(eval_logs, eval_step*config.system.num_updates_per_eval, eval_step*config.system.num_updates_per_eval, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return episode_return


@hydra.main(
    config_path="configs/default",
    config_name="cont_oryx.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.system.task_name = cfg.system.source + "_" + cfg.system.env_name+ "_" + cfg.system.scenario + "_" + cfg.system.dataset

    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}ContOryx experiment completed{Style.RESET_ALL}")



if __name__ == "__main__":
    hydra_entry_point()
