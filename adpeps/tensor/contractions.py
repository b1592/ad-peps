from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, Union

import jax.numpy as np
from yaml import safe_load

from adpeps import ROOT_DIR
from adpeps.utils.empty_tensor import EmptyT
from adpeps.utils.nested import Nested

from .ncon import ncon as st_ncon

f = open(Path(ROOT_DIR, "ipeps", "contractions.yaml"))
ctr = safe_load(f)

TensorType = Union[np.ndarray, Nested, EmptyT]


def ncon(
    tensors: Iterable[TensorType],
    indices_key: Union[str, Iterable[Iterable[int]]],
    **kwargs
) -> TensorType:
    try:
        try:
            ixs = ctr[indices_key]["ix"]
            order = ctr[indices_key].get("order")
        except:
            ixs = ctr[indices_key]
            order = None
        return st_ncon(
            tensors, ixs, empty_class=EmptyT, order=order, mult_method=mult, **kwargs
        )
    except TypeError as e:
        if isinstance(indices_key, str):
            raise e
        return st_ncon(
            tensors, indices_key, empty_class=EmptyT, mult_method=mult, **kwargs
        )


def mult(x: TensorType, y: TensorType, inds: Iterable[int]) -> TensorType:
    if isinstance(x, EmptyT) or isinstance(y, EmptyT):
        return EmptyT()
    elif hasattr(x, "mult"):
        return x.mult(y, inds)
    elif hasattr(y, "mult"):
        x = Nested([x, EmptyT(), EmptyT(), EmptyT()])
        return x.mult(y, inds)
    else:
        return np.tensordot(x, y, inds)
