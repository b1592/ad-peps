""" Main excited-state executable script

    Note:
        The simulations are intended to be used by calling the package 
        directly via :code:`python -m adpeps ...`, as described in 
        :ref:`notes/start`
"""

from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.scipy.optimize import minimize
from jax.test_util import check_grads
from scipy import optimize
from scipy.linalg import eigh, eig
from yaml import safe_load, dump
import jax
import jax.numpy as np
import numpy as onp

from adpeps.ipeps.ipeps import iPEPS, iPEPS_exci
from adpeps.ipeps.make_momentum_path import make_momentum_path
from adpeps.utils import io
from adpeps.utils.printing import print
from adpeps.ipeps.evaluation import filter_null_modes
import adpeps.ipeps.config as sim_config

def run(config_file: str, momentum_ix: int):
    """ Start the simulation

        Args:
            config_file: filename of the configuration file
            momentum_ix: index of the point in momentum space
    """

    print(config_file)
    with open(config_file) as f:
        cfg = safe_load(f)

    # Show options
    print(dump(cfg))

    sim_config.from_dict(cfg)
    base_file = io.get_exci_base_file()
    if not base_file.exists():
        print(f"Base file {base_file} not found. Prepare the simulation first by \
                running with option '-i'")
        return

    sim = iPEPSExciSimulation(config_file, momentum_ix)
    output_folder = io.get_exci_folder()
    output_folder.mkdir(parents=True, exist_ok=True)
    kxs, kys = make_momentum_path(sim_config.momentum_path)
    sim_config.px = kxs[momentum_ix]
    sim_config.py = kys[momentum_ix]
    output_file = io.get_exci_file(momentum_ix)
    print(f"Output: {output_file}", level=2)
    basis_size = sim.basis_size
    res_dtype = np.complex128
    H = onp.zeros((basis_size,basis_size), dtype=res_dtype)
    N = onp.zeros((basis_size,basis_size), dtype=res_dtype)

    for m in range(basis_size):
        grad_H, grad_N = sim(m)
        H[:,m] = grad_H
        N[:,m] = grad_N
        onp.savez(output_file, H=H, N=N)

    print(H)
    print(N)
    onp.savez(output_file, H=H, N=N)
    print('Done')
    print(f"Saved to {output_file}")

def prepare(config_file):
    with open(config_file) as f:
        cfg = safe_load(f)
    sim_config.from_dict(cfg)
    base_file = io.get_exci_base_file()
    print(base_file)
    peps = iPEPS()

    gs_file = io.get_gs_file()
    loaded_sim = np.load(gs_file, allow_pickle=True)
    peps       = loaded_sim['peps'].item()

    sim_config.ctm_max_iter = 30
    sim_config.ctm_conv_tol = 1e-12

    # Converge GS boundary tensors
    peps.converge_boundaries()

    # Convert to excitations iPEPS
    peps.__class__ = iPEPS_exci

    # Normalize the ground-state tensors such that the state has norm 1
    peps.normalize_gs()

    # Shift the Hamiltonian by the ground-state energy 
    # The excited state energy is then relative to the ground state
    peps.substract_gs_energy()

    # Prepare an orthonormal basis with respect to the ground state
    print('Preparing orthonormal basis')
    basis = peps.compute_orth_basis()

    print(f"Saving base to {base_file}")
    np.savez(base_file, peps=peps, basis=basis)

def evaluate_single(config_file, momentum_ix):
    def _compute_ev_red_basis(H, N, P, n):
        P = P[:,:n]
        N2 = P.T.conjugate() @ N @ P
        H2 = P.T.conjugate() @ H @ P
        N2 = 0.5 * (N2 + N2.T.conjugate())
        H2 = 0.5 * (H2 + H2.T.conjugate())
        ev, _ = eig(H2, N2)
        return sorted(ev.real)


    with open(config_file) as f:
        cfg = safe_load(f)

    sim_config.from_dict(cfg)
    kxs, kys = make_momentum_path(sim_config.momentum_path)
    sim_config.px = kxs[momentum_ix]
    sim_config.py = kys[momentum_ix]
    base_file = io.get_exci_base_file()
    base_sim = np.load(base_file, allow_pickle=True)
    output_file = io.get_exci_file(momentum_ix)
    print(output_file)
    dat = np.load(output_file)
    H, N = dat['H'], dat['N']
    basis = base_sim['basis']
    peps  = base_sim['peps'].item()

    # basis = basis.T @ filter_null_modes(peps.tensors, basis)
    # print(basis.shape)
    # print(N.shape)
    # N = basis.T @ N @ basis
    # H = basis.T @ H @ basis
    # H = H.conjugate()

    H = 0.5 * (H + H.T.conjugate())
    N = 0.5 * (N + N.T.conjugate())
    ev_N, P = np.linalg.eig(N)
    idx = ev_N.real.argsort()[::-1]
    ev_N = ev_N[idx]
    selected = (ev_N/ev_N.max()) > 1e-3
    P = P[:,idx]
    P = P[:,selected]
    N2 = P.T.conjugate() @ N @ P
    H2 = P.T.conjugate() @ H @ P
    N2 = 0.5 * (N2 + N2.T.conjugate())
    H2 = 0.5 * (H2 + H2.T.conjugate())
    ev, vectors = eig(H2, N2)
    ixs = np.argsort(ev)
    ev = ev[ixs]
    vectors = vectors[:,ixs]
            
    return sorted(ev.real)

def evaluate(config_file, momentum_ix):
    if momentum_ix != -2:
        return evaluate_single(config_file, momentum_ix)

    with open(config_file) as f:
        cfg = safe_load(f)

    # Show options
    print(dump(cfg))

    sim_config.from_dict(cfg)
    kxs, kys = make_momentum_path(sim_config.momentum_path)

    import matplotlib.pyplot as plt
    evs = []
    for ix in range(len(kxs)):
        try:
            ev = evaluate_single(config_file, ix)
        except:
            ev = [np.nan]
        evs.append(ev[0])
    plt.plot(evs, '--+')
    plt.show()



class iPEPSExciSimulation:
    """ Simulation class for the excited-state simulation 

        Call an instance of this class directly to start the simulation
    """
    def __init__(self, config_file, momentum_ix):
        self.config_file = config_file
        self.momentum_ix = momentum_ix

    @property
    def basis_size(self):
        with open(self.config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)
        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        basis = base_sim['basis']
        return basis.shape[1]

    def __call__(self, ix, v=None):
        print(f"Starting simulation of basis vector {ix+1}/{self.basis_size}")
        with open(self.config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)

        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        basis = np.complex_(base_sim['basis'])
        peps  = base_sim['peps'].item()
        if v is None:
            v = basis[:,ix]
        res, grad_H = value_and_grad(peps.run, has_aux=True)(v)
        grad_H = grad_H.conj()
        print('Res', res, level=2)
        grad_N = res[1].pack_data()
        print('Grad H', grad_H, level=2)
        print('Grad N', grad_N, level=2)
        print(f"========== \nFinished basis vector {ix+1}/{self.basis_size} \n")
        return basis.T @ jax.lax.stop_gradient(grad_H), basis.T @ jax.lax.stop_gradient(grad_N)

    def check_grads(self, A=None):
        with open(self.config_file) as f:
            cfg = safe_load(f)
        sim_config.from_dict(cfg)

        base_file = io.get_exci_base_file()
        base_sim = np.load(base_file, allow_pickle=True)
        basis = np.complex_(base_sim['basis'])
        peps  = base_sim['peps'].item()
        print('Checking gradient')
        # peps.fill(A)
        check_grads(peps.run_gc, (A,), order=1, modes='rev')
        print('Done check')

