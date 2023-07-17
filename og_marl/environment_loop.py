import time
import tensorflow as tf
import tensorflow_io
import numpy as np
import acme
from acme.utils import paths
from og_marl.utils.executor_utils import save_video
import sonnet as snt
from acme.tf import savers as tf2_savers


class EnvironmentLoop:
    """A MARL environment loop (adapted from Mava)"""

    def __init__(
        self,
        environment,
        executor,
        logger,
        record_every=None,
    ):
        # Store executor, logger and environment.
        self._environment = environment
        self._executor = executor
        self._logger = logger

        # Counters
        self._episode_counter = -1
        self._timesteps = 0

        # Recording
        self._record_every = record_every
        if record_every is not None:
            self._fps = 15
            self._fig_size = (600, 600)
            self._frames = []
            self._recordings_path = logger._path("recordings")

    def run_episode(self):
        """Run one episode."""

        # Reset counters
        start_time = time.time()
        episode_steps = 0

        # Reset environment
        timestep, extras = self._environment.reset()

        # Make the first observation.
        self._executor.observe_first(timestep, extras=extras)

        # For evaluation, this keeps track of the total undiscounted reward
        # for each agent accumulated during the episode.
        episode_returns = {}
        rewards = timestep.reward
        for agent, reward in rewards.items():
            episode_returns[agent] = reward

        # Run an episode.
        while not timestep.last():
            # Get agent actions from executor
            actions = self._executor.select_actions(timestep.observation)

            # Step the environment
            timestep, extras = self._environment.step(actions)

            # Maybe record timestep for recording video
            if (
                self._record_every is not None
                and self._episode_counter % self._record_every == 0
            ):
                self._frames.append(self._environment.render(mode="rgb_array"))

            # Have the agents observe the timestep
            self._executor.observe(actions, timestep, extras)

            # Every couple timesteps update the varibale client
            self._executor.update()

            # Book-keeping.
            episode_steps += 1
            rewards = timestep.reward
            for agent, reward in rewards.items():
                episode_returns[agent] += reward

        # Book-keeping.
        self._episode_counter += 1
        self._timesteps += episode_steps

        # Maybe save recording
        if self._record_every is not None and self._frames:
            save_video(
                self._recordings_path,
                self._frames,
                self._fps,
                self._fig_size,
                self._episode_counter,
            )
            self._frames = []

        # Collect the results for logging
        steps_per_second = episode_steps / (time.time() - start_time)
        logs = {
            "episode_length": episode_steps,
            "episode_return": np.mean(list(episode_returns.values())),
            "steps_per_second": steps_per_second,
            "episodes": self._episode_counter,
            "timesteps": self._timesteps
        }

        # Add executor stats to logs
        logs.update(self._executor.get_stats())

        # Add environment stats to logs
        logs.update(self._environment.get_stats())

        # Write logs
        if self._logger is not None:
            self._logger.write(logs)

        return logs

    def run(self):
        """Perform the run loop. Used when run on seperate process."""

        while True:
            logs = self.run_episode()


class EvaluationEnvironmentLoop:
    """Environmentloop purposfully built for evaluation."""
    def __init__(
        self,
        environment,
        executor,
        trainer,
        logger,
        evaluation_period=100,
        evaluation_episodes=32,
        record_every=None,
    ):

        # Internalize executor, trainer, logger and environment.
        self._environment = environment
        self._executor = executor
        self._trainer = trainer
        self._logger = logger

        # Counters
        self._episode_counter = -1
        self._timesteps = 0

        # Evaluation parameters
        self._evaluation_period = evaluation_period
        self._evaluation_episodes = evaluation_episodes
        self._last_evaluation_episode = 0

        # Recording
        self._record_every = record_every
        if record_every is not None:
            self._fps = 30
            self._fig_size = (600, 600)
            self._frames = []
            self._recordings_path = logger._path("recordings")

        # Checkpointing
        self._current_best_return = -float("inf")

    def run(self):
        """Perform the run loop. Used when run on seperate process."""

        while True:
            trainer_steps = self._trainer.get_steps()

            if ((
                trainer_steps - self._last_evaluation_episode # every couple trainer steps, do evaluation
            ) >= self._evaluation_period) or self._last_evaluation_episode == 0:

                evaluation_logs = self.run_evaluation(
                    trainer_steps, self._evaluation_episodes
                )
                self._last_evaluation_episode = trainer_steps
            else:
                time.sleep(0.1)  # sleep before trying to evaluate again

    def run_evaluation(
        self, trainer_steps, use_best_checkpoint=False
    ):
        """Run evaluation"""

        # Get latest variables from trainer
        self._executor.update()

        # Maybe restore best checkpoint
        if use_best_checkpoint and self._executor._must_checkpoint:
            self._executor.restore_checkpoint()

        if hasattr(self._environment, "battles_won"): # SMAC
            old_battles_won = self._environment.battles_won

        # Bookkeeping
        evaluation_episode_returns = []
        evaluation_episode_lengths = []
        completions = []
        scores = []
        for episode in range(self._evaluation_episodes):

            # Reset any counter
            episode_steps = 0

            # Reset environment
            timestep, extras = self._environment.reset()

            # Make the first observation.
            self._executor.observe_first(timestep, extras=extras)

            # For evaluation, this keeps track of the total undiscounted reward
            # for each agent accumulated during the episode.
            episode_returns = {}
            rewards = timestep.reward
            for agent, reward in rewards.items():
                episode_returns[agent] = reward

            # Run an episode.
            while not timestep.last():

                actions = self._executor.select_actions(timestep.observation)

                # Step the environment
                timestep, extras = self._environment.step(actions)

                # Maybe record timestep for recording
                if (
                    self._record_every is not None
                    and self._episode_counter % self._record_every == 0
                ):
                    self._frames.append(self._environment.render(mode="rgb_array"))

                # Have the agent observe the timestep and let the actor update itself.
                self._executor.observe(actions, timestep, extras)

                # Book-keeping.
                episode_steps += 1

                rewards = timestep.reward
                for agent, reward in rewards.items():
                    episode_returns[agent] += reward

            env_stats = self._environment.get_stats()

            if "completion" in env_stats:
                completions.append(env_stats["completion"])
                scores.append(env_stats["score"])

            # Book-keeping.
            self._episode_counter += 1

            # Maybe save recording
            if self._record_every is not None and self._frames:
                save_video(
                    self._recordings_path,
                    self._frames,
                    self._fps,
                    self._fig_size,
                    self._episode_counter,
                )
                self._frames = []

            # Store episode return and length
            evaluation_episode_returns.append(np.mean(list(episode_returns.values()))) # mean over agents
            evaluation_episode_lengths.append(episode_steps)

        # Report mean over all evaluation runs
        average_return = np.mean(evaluation_episode_returns)
        average_episode_lens = np.mean(evaluation_episode_lengths)
        evaluation_logs = {
            "evaluator_episode_return": average_return,
            "trainer_steps": trainer_steps,
            "evaluator_episode_length": average_episode_lens,
        }

        # Checkpoint if new best policy
        if completions: # FLATLAND
            evaluation_logs["completion_rate"] = np.mean(completions)
            mean_score = np.mean(scores)
            evaluation_logs["score"] = mean_score
            self._checkpoint_best(
                mean_score, timestep
            )  # in FLATLAND use score for checkpointing
        else:
            self._checkpoint_best(average_return)  # else use return

        if hasattr(self._environment, "battles_won"): # SMAC
            win_rate = (
                self._environment.battles_won - old_battles_won
            ) / self._evaluation_episodes
            evaluation_logs.update({"evaluator_win_rate": win_rate})

        # Log the given results.
        if self._logger is not None:
            self._logger.write(evaluation_logs)

        return evaluation_logs

    def _checkpoint_best(self, current_return):
        """Checkpoint if the new best policy."""

        trainer_steps = self._trainer.get_steps()
        if self._executor._must_checkpoint:
            if trainer_steps > 0 and current_return > self._current_best_return:
                self._current_best_return = current_return
                self._executor.checkpoint()
