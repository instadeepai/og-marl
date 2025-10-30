from typing import Any, Dict, Protocol, Tuple

import chex
import jumanji.specs as specs
from jumanji.types import TimeStep
from typing_extensions import TypeAlias

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any
Metrics: TypeAlias = Dict[str, chex.Array]

class MarlEnv(Protocol):
    """The API used by mava for environments.

    A mava environment simply uses the Jumanji env API with a few added attributes.
    For examples of how to add custom environments to Mava see `mava/wrappers/jumanji.py`.
    Jumanji API docs: https://instadeepai.github.io/jumanji/#basic-usage
    """

    num_agents: int
    time_limit: int
    action_dim: int

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Resets the environment to an initial state.

        Args:
            key: random key used to reset the environment.

        Returns:
            state: State object corresponding to the new state of the environment,
            timestep: TimeStep object corresponding the first timestep returned by the environment,
        """
        ...

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Run one timestep of the environment's dynamics.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the action to take.

        Returns:
            state: State object corresponding to the next state of the environment,
            timestep: TimeStep object corresponding the timestep returned by the environment,
        """
        ...

    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec.

        Returns:
            observation_spec: a NestedSpec tree of spec.
        """
        ...

    def action_spec(self) -> specs.Spec:
        """Returns the action spec.

        Returns:
            action_spec: a NestedSpec tree of spec.
        """
        ...

    def reward_spec(self) -> specs.Array:
        """Describes the reward returned by the environment. By default, this is assumed to be a
        single float.

        Returns:
            reward_spec: a `specs.Array` spec.
        """
        ...

    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment. By default, this is assumed to be a
        single float between 0 and 1.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        ...

    @property
    def unwrapped(self) -> Any:
        """Retuns: the innermost environment (without any wrappers applied)."""
        ...
