""" Configuration module for iPEPS simulations 

    These settings will be loaded from a configuration file :code:`.yml` file 
    via the :meth:`from_dict` function
"""

import math
import os
from typing import Iterable, Union

import numpy as np

D: int = None
""" iPEPS bond dimension """

chi: int = None
""" CTM boundary bond dimension """

model: str = None
""" Model """

model_params: dict = None
""" Model parameters """

method: str = None
""" Optimization method """

seed: int = 1
""" Random seed for initial state """

resume: bool = False
""" Resume earlier simulation if found """

base_sim: Union[str, None] = None
""" Base simulation """

load_sim: bool = None
""" Load previous (Python) simulation """

max_iterations: int = 100
""" Maximum number of optimizer iterations """

disp_level: int = 1
""" Display level (`0`: no output) """

pattern: Union[Iterable, None] = None
""" Unit cell configuration
    Defined as a 2-D array of integers that label the unique sites in the unit 
    cell.
    
    Example:
        A 2x2 unit cell with a [AB, BA]-type pattern is defined by

        .. code-block:: python

            pattern            = [
                [0, 1],
                [1, 0]
            ]
"""

ctm_conv_tol: float = 1e-10
""" CTM convergence criterium (singular values norm difference) """

ctm_min_iter: int = 5
""" Minimal number of CTM steps """

ctm_max_iter: int = 20
""" Maximal number of CTM steps """

flush_output: bool = False
""" Passes the :code:`flush    = True` argument to the builtin :code:`print` function
    when calling the :func:`adpeps.utils.printing.print` function

    Useful when deploying the code to computing clusters and capturing the output 
    into text files
"""

out_prefix: str = ""
""" Optional prefix for the output file of the simulation
    
    Example:
        :code:`.../{model}_D{D}_X{chi}.npz`

        becomes

        :code:`.../{out_prefix}_{model}_D{D}_X{chi}.npz`
"""

# Excitation settings

px: float = 0 * math.pi

py: float = 0 * math.pi

momentum_path: str = "Bril1"
""" Momentum path through the BZ """

filter_null: bool = False


def from_dict(cfg):
    """Import config from configuration (`.yml`) file"""

    cfg_vars = globals()
    for name, value in cfg.items():
        if name in cfg_vars.keys():
            cfg_vars[name] = value
        else:
            raise ValueError(f"Option {name} = {value} not defined in iPEPS config")
    try:
        debug_override = os.environ.get("PY_SIM_DEBUG2")
        if debug_override is not None and int(debug_override) == 1:
            print("** Debug mode on (PY_SIM_DEBUG = 1) **")
            cfg_vars["disp_level"] = 2
            cfg_vars["use_parallel"] = False
    except:
        pass


def get_model_params():
    if model_params is not None:
        try:
            return "_".join([str(p) for p in model_params.values()]) + "_"
        except AttributeError:
            return "_".join([str(p) for p in model_params]) + "_"
    else:
        return ""
