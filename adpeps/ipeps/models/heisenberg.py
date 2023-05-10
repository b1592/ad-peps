""" 2D Heisenberg model """

import jax.numpy as np

import adpeps.ipeps.config as sim_config
from adpeps.utils.tlist import set_pattern

from .common import sigmam, sigmap, sigmaz

name = "Heisenberg spin-1/2 model"


def setup():
    """Returns the Hamiltonian"""
    H = make_hamiltonian(**sim_config.model_params)

    obs = None
    return H, obs


def make_hamiltonian(J=1):
    """Heisenberg model"""
    H = (
        tprod(sigmaz, sigmaz) / 4
        + tprod(sigmap, sigmam) / 2
        + tprod(sigmam, sigmap) / 2
    )
    H = J * H
    return H


def tprod(a, b):
    return np.outer(a, b).reshape([2, 2, 2, 2], order="F").transpose([0, 2, 1, 3])
