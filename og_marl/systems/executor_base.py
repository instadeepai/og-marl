import tensorflow as tf
from acme.tf import savers as tf2_savers


class ExecutorBase:
    def __init__(
        self,
        agents,
        variable_client,
        add_agent_id_to_obs=False,
        checkpoint_subpath="",
        must_checkpoint=False,
    ):
        # List of agent keys
        self._agents = agents

        # Variable client to get new variables from trainer
        self._variable_client = variable_client

        # Concat agent IDs to observation
        self._add_agent_id_to_obs = add_agent_id_to_obs

        # Checkpointing
        self._must_checkpoint = must_checkpoint
        self._checkpoint_path = checkpoint_subpath
        self._variables_to_checkpoint = {}

        if self._must_checkpoint:
            self.restore_checkpoint()

    def restore_checkpoint(self):
        "Setup and and restore model checkpoints."

        self._checkpointer = tf2_savers.Checkpointer(
            directory=self._checkpoint_path,
            objects_to_save=self._variables_to_checkpoint,
            time_delta_minutes=1.0,
            add_uid=False,
            max_to_keep=1,
        )

    def checkpoint(self):
        "Save model checkpoints."

        self._checkpointer.save(force=True)

    def update(self, wait=False):
        """Update executor variables."""

        if self._variable_client:
            self._variable_client.update(wait)

    def observe_first(self, timestep, extras={}):
        raise NotImplementedError

    def observe(self, actions, next_timestep, next_extras):
        raise NotImplementedError

    def select_actions(self, observations):
        raise NotImplementedError

    @tf.function
    def _select_actions(self, observations):
        raise NotImplementedError

    def hook_after_action_selection(self, time_t):
        """After action selection hook."""
        pass

    def get_stats(self):
        """Return extra executor stats to log."""
        return {}
