""" Utility module for printing output depending on the verbosity setting 
    :attr:`adpeps.ipeps.config.disp_level` in the configuration file.
"""

import builtins
import time

import adpeps.ipeps.config as sim_config

prefix = None
show_time = False


def print(*args, level: int = None, **kwargs):
    """Print output using builtin :code:`print` if :code:`level`
    <= :attr:`adpeps.ipeps.config.disp_level`

    Args:
        *args: arbitraty arguments to be passed to builtin :code:`print`
        level: verbosity level, determining at which verbosity setting this
            should be printed
        **kwargs: arbitraty keyword arguments for builtin :code:`print`
    """
    if level is None or level <= sim_config.disp_level:
        if sim_config.flush_output:
            kwargs["flush"] = True
        if prefix is not None:
            if show_time:
                curtime = time.strftime("[%H:%M:%S]", time.localtime())
                builtins.print(prefix, curtime, *args, **kwargs)
            else:
                builtins.print(prefix, *args, **kwargs)
        else:
            if show_time:
                curtime = time.strftime("[%H:%M:%S]", time.localtime())
                builtins.print(curtime, *args, **kwargs)
            else:
                builtins.print(*args, **kwargs)
