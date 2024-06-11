import tensorflow as tf
from tensorflow import Tensor
import sonnet as snt


@snt.allow_empty_variables
class IdentityNetwork(snt.Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def __call__(self, x: Tensor) -> Tensor:
        return x