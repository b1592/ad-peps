""" IO module with convenience function for forming the localized filenames 
    and foldernames of the relevant configuration files and data files.

    Note:
        The input configuration file location can be set via the 
        :envvar:`CONFIGDIR` variable. If it is not set, the default 
        input folder will be the `examples` subfolder of the package 
        root directory

    Note:
        The output data location can be set via the :envvar:`DATADIR` 
        variable. If it is not set, the default output folder will be 
        in the `simulations` subfolder of the package root directory.
"""

from pathlib import Path
import math
import numpy as np
import os

import adpeps
import adpeps.ipeps.config as sim_config


def localize_data_file(filename):
    ROOT_DIR = adpeps.PROJECT_ROOT_DIR
    try:
        base_out_folder = os.environ["DATADIR2"]
    except KeyError:
        base_out_folder = Path(ROOT_DIR, 'simulations')
    return Path(base_out_folder, filename)

def localize_config_file(filename):
    ROOT_DIR = adpeps.PROJECT_ROOT_DIR
    try:
        base_out_folder = os.environ["CONFIGDIR"]
    except KeyError:
        base_out_folder = Path(ROOT_DIR, 'examples')
    return Path(base_out_folder, filename).with_suffix('.yaml')

def get_gs_file():
    if sim_config.out_prefix is not None:
        filename = f"{sim_config.out_prefix}_{sim_config.model}_D{sim_config.D}_X{sim_config.chi}"
    else:
        filename = f"{sim_config.model}_D{sim_config.D}_X{sim_config.chi}"
    filename = Path('gs', filename)
    return localize_data_file(filename).with_suffix('.npz')

def get_exci_folder():
    if sim_config.out_prefix is not None:
        folder = f"{sim_config.out_prefix}_{sim_config.model}_D{sim_config.D}_X{sim_config.chi}"
    else:
        folder = f"{sim_config.model}_D{sim_config.D}_X{sim_config.chi}"
    folder = Path('exci', folder)
    return localize_data_file(folder)

def get_exci_file(momentum_ix):
    foldername = get_exci_folder()
    filename = f"{momentum_ix+1}_{sim_config.px/math.pi:.5}_{sim_config.py/math.pi:.5}.npz"
    return Path(foldername, filename)

def get_exci_base_file():
    return get_exci_folder().with_suffix('.base.npz')

