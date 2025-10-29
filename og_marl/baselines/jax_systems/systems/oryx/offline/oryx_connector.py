import copy
from functools import partial
import os
from typing import Any, Callable, Dict, Tuple, NamedTuple
import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flashbax.vault import Vault
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from flashbax.buffers.flat_buffer import TrajectoryBuffer, TrajectoryBufferState
from og_marl.baselines.jax_systems.evaluator import ActorState, EvalActFn, get_eval_fn, get_num_eval_envs
from og_marl.baselines.jax_systems.networks.oryx_network import OryxNetwork
from og_marl.baselines.jax_systems.networks.utils.oryx import get_init_hidden_state
from og_marl.baselines.jax_systems.systems.oryx.types import (
    ActorApply,
    LearnerApply,
    Params,
    Transition,
    OptState,
)
from og_marl.baselines.jax_systems.env_types import MarlEnv
from og_marl.baselines.jax_systems.types import Action, ExperimentOutput, LearnerFn, Metrics
from og_marl.baselines.jax_systems.utils import make_env as environments
from og_marl.baselines.jax_systems.utils.logger import LogEvent, MavaLogger
from og_marl.baselines.jax_systems.utils.network_utils import get_action_head
from og_marl.baselines.jax_systems.utils.training import make_learning_rate
from og_marl.baselines.jax_systems.types import Observation
                                              
class LearnerState(NamedTuple):
    """State of the offline learner for Memory Sable"""

    params: Params
    opt_states: OptState
    key: Any
    steps: int


def get_learner_fn(
    apply_fns: Tuple[ActorApply, LearnerApply],
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply functions for executing and training the network.
    _, apply_fn = apply_fns

    def _update_step(learner_state: LearnerState, batch: Any) -> Tuple[LearnerState, Tuple[Transition, Metrics]]:
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
            batch (Tuple): The current batch

        """

        # UNPACK NEW LEARNER STATE
        (
            params,
            opt_states,
            key,
            steps,
        ) = learner_state

        def _update_epoch(train_state: Tuple, batch: Tuple) -> Tuple:
            params, opt_states, key = train_state

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
                    
                    _, target_q_values = apply_fn(  # type: ignore
                        target_params,
                        observation=batch.obs,
                        action=batch.action,
                        dones=batch.done,
                        hstates=get_init_hidden_state(
                            config.network.net_config, B
                        ),
                    )

                    # Select target q values according to online predicted actions.
                    target_next_action_value = jnp.squeeze(
                        jnp.take_along_axis(
                            target_q_values,
                            jnp.expand_dims(batch.action, axis=-1).astype(jnp.int32),
                            axis=-1,
                        ),
                        axis=-1,
                    )

                    advantage_q = jax.nn.softmax(target_next_action_value / config.system.value_temperature, axis=0) # across batch dim
                    target_next_action_value = advantage_q.shape[0] * advantage_q * target_next_action_value

                    target = batch.reward[:,:-N] + (
                        config.system.gamma * (1 - batch.done[:,N:]) * target_next_action_value[:,N:]
                    )

                    target = jax.lax.stop_gradient(target)

                    # TD Error
                    td_error = 0.5 * jnp.square(target - action_q_value[:,:-N]).mean()

                    # Policy Loss
                    policy_probs = jax.nn.softmax(logits, axis=-1)
                    state_value = jnp.sum(policy_probs * q_values, axis=-1)
                    advantage = jax.lax.stop_gradient(action_q_value - state_value)

                    advantage_weight = jax.lax.stop_gradient(
                        jax.nn.softmax(advantage / config.system.policy_temperature, axis=0)
                    ) * len(advantage)

                    action_prob = jnp.take_along_axis(
                        policy_probs,
                        jnp.expand_dims(batch.action, axis=-1).astype(jnp.int32),
                        axis=-1,
                    )
                    
                    # Compute weighted policy loss
                    policy_loss = jnp.mean(-1 * advantage_weight * jnp.log(action_prob).squeeze(-1))

                    loss =  config.system.critic_coef * td_error + policy_loss


                    return loss, (td_error, policy_loss, action_q_value.mean(), advantage.mean())

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
                    "coma_loss": loss_info[1][1],
                    "q_values": loss_info[1][2],
                    "advantage": loss_info[1][3]
                }

                return (new_params, new_opt_state, key), loss_info

            # Sample a batch of trajectories
            key, agent_shuffle_key, training_key = jax.random.split(key, 3)

            batch = batch.experience
            batch = Transition(
                obs=Observation(
                    agents_view=batch["obs"]["agents_view"],
                    action_mask=batch["obs"]["action_mask"],
                    step_count=batch["obs"]["step_count"],
                ),
                action=batch["action"],
                reward=batch["reward"],
                done=batch["done"],
                done_mask=batch["done_mask"], 
            )

            # Shuffle agents
            agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
            batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=2), batch)  # NBNB

            # Update network according to the batch
            (params, opt_states, _), loss_info = _update((params, opt_states, training_key), batch)

            update_state = (
                params,
                opt_states,
                key,
            )
            return update_state, loss_info

        update_state = (params, opt_states, key)

        # UPDATE EPOCHS
        update_state, loss_info = _update_epoch(update_state, batch)
        params, opt_states, key = update_state

        learner_state = LearnerState(
            params,
            opt_states,
            key,
            steps+1,
        )
        return learner_state, loss_info

    def learner_fn(learner_state: LearnerState, batch: Tuple) -> ExperimentOutput[LearnerState]:
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
        learner_state, loss_info = _update_step(learner_state, batch)
        
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics={},
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Callable, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, net_key = keys

    # Get number of agents and actions.
    action_dim = env.action_dim
    n_agents = env.num_agents
    config.system.num_agents = n_agents
    config.system.num_actions = action_dim

    config.network.memory_config.chunk_size = config.system.sample_sequence_length * n_agents

    _, action_space_type = get_action_head("discrete")

    # Define network.
    oryx_network = OryxNetwork(
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
    init_obs = env.observation_spec().generate_value()
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)  # Add batch dim
    init_hs = get_init_hidden_state(config.network.net_config, batch_size=1)

    # Initialise params and optimiser state.
    online_params = oryx_network.init(
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
        partial(
            oryx_network.apply, method="get_actions"
        ),  # Execution function
        partial(oryx_network.apply),  # Training function
    )
    eval_apply_fn = partial(oryx_network.apply, method="get_actions")

    if config.env.scenario.task_name in ["con-18x18x30a", "con-22x22x40a", "con-25x25x50a"]:
        with jax.default_device(jax.devices("cpu")[0]):
            vlt = Vault("oryx/connector", vault_uid=config.env.scenario.task_name)
            
            rb = fbx.make_trajectory_buffer(
                sample_sequence_length=config.system.sample_sequence_length,
                period=1,
                sample_batch_size=config.system.sample_batch_size,
                add_batch_size=1, # dummy since we dont add data to buffer
                max_length_time_axis=424242, # dummy
                min_length_time_axis=424242, # dummy
            )

            buffer_state = vlt.read()
    else:    
        vlt = Vault("oryx/connector", vault_uid=config.env.scenario.task_name)
        
        rb = fbx.make_trajectory_buffer(
            sample_sequence_length=config.system.sample_sequence_length,
            period=1,
            sample_batch_size=config.system.sample_batch_size,
            add_batch_size=1, # dummy since we dont add data to buffer
            max_length_time_axis=424242, # dummy
            min_length_time_axis=424242, # dummy
        )

        buffer_state = vlt.read()

    learn = get_learner_fn(apply_fns, optim.update, config)

    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)

    init_learner_state = LearnerState(
        params=params,
        opt_states=opt_state,
        key=step_keys,
        steps=0,
    )

    return learn, eval_apply_fn, init_learner_state, rb, buffer_state, config


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "Oryx"
    _config.system_name = "Oryx"
    config = copy.deepcopy(_config)

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.system.num_evaluation

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=3)

    # Setup learner.
    learn, oryx_execution_fn, learner_state, rb, buffer_state, config = learner_setup(env, (key, net_key), config)
    learn = jax.jit(chex.assert_max_traces(learn, 1))
    sample_fn = jax.jit(rb.sample)

    # Setup evaluator.
    def make_oryx_act_fn(actor_apply_fn: ActorApply) -> EvalActFn:
        _hidden_state = "hidden_state"

        def eval_act_fn(
            params: Params, timestep: TimeStep, key: chex.PRNGKey, actor_state: ActorState
        ) -> Tuple[Action, Dict]:
            hidden_state = actor_state[_hidden_state]
            output_action, _, _, hidden_state = actor_apply_fn(  # type: ignore
                params,
                timestep.observation,
                hidden_state,
                key=key,
            )
            return output_action, {_hidden_state: hidden_state}

        return eval_act_fn

    # One key per device for evaluation.
    key_e, eval_key = jax.random.split(key_e)

    # Define Apply fn for evaluation.
    eval_act_fn = make_oryx_act_fn(oryx_execution_fn)

    # Create evaluator
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    pprint(cfg)
        
    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)

    # First Evaluation
    trained_params = learner_state.params.online
    eval_metrics = evaluator(trained_params, eval_key, {"hidden_state": eval_hs})
    eval_step = 0
    logger.log(eval_metrics, eval_step, eval_step, LogEvent.EVAL)

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None

    # generate_samples
    if config.env.scenario.task_name in ["con-18x18x30a", "con-22x22x40a", "con-25x25x50a"]:
        with jax.default_device(jax.devices("cpu")[0]):
            batch = sample_fn(buffer_state, key)
    else:
        batch = sample_fn(buffer_state, key)

    for eval_step in range(config.system.num_evaluation):

        for _ in range(config.system.num_updates_per_eval):
            # Train.
            learner_output = learn(learner_state, batch)
            learner_state = learner_output.learner_state
            
            if config.env.scenario.task_name in ["con-18x18x30a", "con-22x22x40a", "con-25x25x50a"]:
                with jax.default_device(jax.devices("cpu")[0]):
                    key, rb_key = jax.random.split(key)
                    batch = sample_fn(buffer_state, rb_key)
            else:
                key, rb_key = jax.random.split(key)
                batch = sample_fn(buffer_state, rb_key)

        t_steps = (eval_step+1) * config.system.num_updates_per_eval
        training_step = {"training_updates": t_steps}
        logger.log(training_step, t_steps, t_steps, LogEvent.MISC)

        # Log the results of the training.
        logger.log(learner_output.train_metrics, t_steps, t_steps, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = learner_output.learner_state.params.online
        key_e, eval_key = jax.random.split(key_e)

        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_key, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t_steps, t_steps, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

    # Measure absolute metric.
    eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
    abs_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
    abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
    key, eval_key = jax.random.split(key)
    params = best_params
    eval_metrics = abs_metric_evaluator(params, eval_key, {"hidden_state": abs_hs})

    logger.log(eval_metrics, eval_step, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return episode_return


@hydra.main(
    config_path="../../../configs/default",
    config_name="oryx_connector.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Oryx Connector experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
