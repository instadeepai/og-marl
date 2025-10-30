from typing import Callable, Dict, Tuple, Any

from chex import Array, PRNGKey
from flashbax.buffers.flat_buffer import TrajectoryBufferState
from flax.core.frozen_dict import FrozenDict
from optax._src.base import OptState
from typing_extensions import NamedTuple


class SableNetworkConfig(NamedTuple):
    """Configuration for the Sable network."""

    n_block: int
    n_head: int
    embed_dim: int


class HiddenStates(NamedTuple):
    """Hidden states for the encoder and decoder."""

    encoder: Array
    decoder_self_retn: Array
    decoder_cross_retn: Array


class Params(NamedTuple):
    """Parameters for online and target networks."""

    online: FrozenDict
    target: FrozenDict


class RecLearnerState(NamedTuple):
    """State of the learner for Memory Sable"""

    params: Params
    opt_states: OptState
    key: PRNGKey
    env_state: Array
    timestep: Any
    buffer_state: TrajectoryBufferState
    n_env_steps: Array
    hstates: HiddenStates

class Transition(NamedTuple):
    """Transition tuple."""

    done: Array
    action: Array
    reward: Array
    obs: Array
    done_mask: Array = None
    train_mask: Array = None
    info: Dict = None


ActorApply = Callable[
    [FrozenDict, Array, Array, HiddenStates, PRNGKey],
    Tuple[Array, Array, Array, Array, HiddenStates],
]
LearnerApply = Callable[
    [FrozenDict, Array, Array, Array, HiddenStates, Array, PRNGKey], Tuple[Array, Array, Array]
]
