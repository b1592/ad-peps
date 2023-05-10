import cmath

import jax.numpy as np
import numpy as onp
import scipy.linalg as linalg
from jax import random

import adpeps.ipeps.config as sim_config
from adpeps.tensor.contractions import ncon
from adpeps.utils.empty_tensor import EmptyT
from adpeps.utils.nested import Nested
from adpeps.utils.printing import print
from adpeps.utils.tlist import TList, cur_loc, set_pattern

"""
    Evaluation module for iPEPS simulations

    This module contains the contractions of the reduced density matrices 
    and the computation of the expectation values for iPEPS ground- and 
    excited states
"""


def get_gs_energy(H, tensors):
    """Returns ground-state energy and norm of the iPEPS"""
    E, nrm, *_ = get_obs(H, tensors, measure_obs=False)
    return E[0], nrm


def get_all_energy(H, tensors):
    """Returns only energy and norm of the iPEPS"""
    E, nrm, _ = get_obs(H, tensors, measure_obs=False)
    return E


def get_obs(H, tensors, measure_obs=True, only_gs=False):
    """Returns the energy and norm of the state

    The energy will be returned as a `Nested` tensor

    More observables can be added here
    """
    A = tensors.A
    Ad = tensors.Ad
    Ehs = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    Evs = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    Ehs_exci = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    Evs_exci = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    nrmhs = TList(shape=A.size, pattern=A.pattern)  # Horizontal terms
    nrmvs = TList(shape=A.size, pattern=A.pattern)  # Vertical terms
    obs_evs = [TList(shape=A.size, pattern=A.pattern) for _ in tensors.observables]

    for i in A.x_major():
        with cur_loc(i):
            if not Evs.is_changed(0, 0):
                roh, rov = get_dms(tensors)

                nrmh = np.trace(np.reshape(roh[0], (4, 4))).real
                nrmv = np.trace(np.reshape(rov[0], (4, 4))).real
                nrmhs[0, 1] = nrmh
                nrmvs[0, 0] = nrmv
                roh = roh / nrmh
                rov = rov / nrmv

                Ehs[0, 1] = ncon([roh, H], ([1, 2, 3, 4], [1, 2, 3, 4])).real
                Evs[0, 0] = ncon([rov, H], ([1, 2, 3, 4], [1, 2, 3, 4])).real

                # if measure_obs:
                #     ro_one = get_one_site_dm(tensors.Cs,tensors.Ts,A,Ad)
                #     for obs_i,obs in enumerate(tensors.observables):
                #         if obs.size == 1:
                #             try:
                #                 obs_ev = ncon([ro_one, obs.operator], ([1,2],[1,2]))
                #                 print(f"Obs {(obs_i,i)} {obs.__repr__()}: {obs_ev.item()}", level=2)
                #                 obs_evs[obs_i][0,0] = obs_ev.item()
                #             except:
                #                 obs_evs[obs_i][0,0] = np.nan
                #         elif obs.size == 2:
                #             try:
                #                 obs_ev_h = ncon([roh, obs.operator], ([1,2,3,4],[1,2,3,4]))
                #                 obs_ev_v = ncon([rov, obs.operator], ([1,2,3,4],[1,2,3,4]))
                #                 print(f"Obs {(obs_i,i)} {obs.__repr__()}: {obs_ev_h.item()}, {obs_ev_v.item()}", level=2)
                #                 obs_evs[obs_i][0,0] = (obs_ev_h.item(), obs_ev_v.item())
                #             except:
                #                 obs_evs[obs_i][0,0] = (np.nan, np.nan)

    # try:
    #     print(Ehs.mean(), Evs.mean(), level=2)
    # except:
    #     print(Ehs.mean(), Evs.mean(), level=2)
    E = Ehs.mean() + Evs.mean()
    nrm = 0.5 * (nrmhs.mean() + nrmvs.mean())
    return E, nrm, obs_evs


def compute_exci_norm(tensors):
    """Returns the norm of the excited state based on a one-site
    environment

    Averaged over sites in the unit cell
    """
    A = tensors.A
    nrms = TList(shape=A.size, pattern=A.pattern)
    nrms_gs = TList(shape=A.size, pattern=A.pattern)
    envBs = TList(shape=A.size, pattern=A.pattern)

    for i in A.x_major():
        with cur_loc(i):
            if not nrms.is_changed(0, 0):
                nrm, nrm_gs, envB = _compute_one_site_exci_norm(tensors)
                # Exci norm
                nrms[0, 0] = nrm
                # Ground state norm
                nrms_gs[0, 0] = nrm_gs
                # Environment (exci norm without center Bd)
                envBs[0, 0] = envB
    return nrms.mean(), nrms_gs.mean(), envBs, nrms_gs


def _compute_one_site_exci_norm(ts):
    """Returns the norm of the excited state for one site in the
    unit cell
    """

    def get_single_site_dm(C1, T1, C2, T2, C3, T3, C4, T4):
        return ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")

    n_tensors = [
        ts.Cs[0][-1, -1],
        ts.Ts[0][0, -1],
        ts.Cs[1][1, -1],
        ts.Ts[1][1, 0],
        ts.Cs[2][1, 1],
        ts.Ts[2][0, 1],
        ts.Cs[3][-1, 1],
        ts.Ts[3][-1, 0],
    ]
    B_tensors = [
        ts.B_Cs[0][-1, -1],
        ts.B_Ts[0][0, -1],
        ts.B_Cs[1][1, -1],
        ts.B_Ts[1][1, 0],
        ts.B_Cs[2][1, 1],
        ts.B_Ts[2][0, 1],
        ts.B_Cs[3][-1, 1],
        ts.B_Ts[3][-1, 0],
    ]
    Bd_tensors = [
        ts.Bd_Cs[0][-1, -1],
        ts.Bd_Ts[0][0, -1],
        ts.Bd_Cs[1][1, -1],
        ts.Bd_Ts[1][1, 0],
        ts.Bd_Cs[2][1, 1],
        ts.Bd_Ts[2][0, 1],
        ts.Bd_Cs[3][-1, 1],
        ts.Bd_Ts[3][-1, 0],
    ]

    # Compute the ground state one-site reduced density matrix
    n_dm = get_single_site_dm(*n_tensors)
    nrm0 = ncon(
        (ts.A[0, 0], ts.Ad[0, 0], n_dm),
        ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
    )

    B_dm = EmptyT()
    for i in range(8):
        # Start with all regular (ground state) boundary tensors
        cur_tensors = n_tensors.copy()
        cur_tensors[i] = B_tensors[i]
        # Compute the one-site reduced density matrix and add it to the
        # total
        new_dm = get_single_site_dm(*cur_tensors)
        B_dm = B_dm + new_dm

    # The full norm can be split into two parts:
    #   - One B and Bd on the same center site, with regular boundary tensors
    #   - One Bd in the center and a B in the boundaries (many terms)
    nrm_exci = (
        ncon(
            (ts.B[0, 0], ts.Bd[0, 0], n_dm),
            ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
        )
        + ncon(
            (ts.A[0, 0], ts.Bd[0, 0], B_dm),
            ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
        )
    ) / nrm0

    # The row of the norm overlap matrix (i.e. the gradient of the norm) is the
    # reduced density matrix contracted with only the ket-layer of the center site
    nrmB_open = (
        ncon((ts.B[0, 0], n_dm), ([-1, 2, 3, 4, 5], [2, 3, 4, 5, -2, -3, -4, -5]))
        + ncon((ts.A[0, 0], B_dm), ([-1, 2, 3, 4, 5], [2, 3, 4, 5, -2, -3, -4, -5]))
    ) / nrm0

    try:
        print("B norm", nrm_exci.item(), " | Gs norm", nrm0.item(), level=1)
    except:
        pass
    return nrm_exci.real, nrm0, nrmB_open


def get_orth_basis(tensors):
    """Returns a basis of vectors orthogonal to the ground state

    Each of these vectors can be used as an input for the iPEPS
    excitation object
    """

    def get_single_site_dm(C1, T1, C2, T2, C3, T3, C4, T4):
        return ncon((C2, T1, C1, T4, C4, T3, C3, T2), "dm_single_site")

    basis = None
    A = tensors.A
    Ad = tensors.Ad
    nrms = TList(shape=A.size, pattern=A.pattern)
    for i in A.x_major():
        with cur_loc(i):
            if not nrms.is_changed(0, 0):
                n_tensors = [
                    tensors.Cs[0][-1, -1],
                    tensors.Ts[0][0, -1],
                    tensors.Cs[1][1, -1],
                    tensors.Ts[1][1, 0],
                    tensors.Cs[2][1, 1],
                    tensors.Ts[2][0, 1],
                    tensors.Cs[3][-1, 1],
                    tensors.Ts[3][-1, 0],
                ]
                # Compute the ground state one-site reduced density matrix
                n_dm = get_single_site_dm(*n_tensors)
                nrm0 = ncon(
                    (tensors.A[0, 0], tensors.Ad[0, 0], n_dm),
                    ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9]),
                )
                nrms[0, 0] = nrm0
                env_0 = ncon(
                    (tensors.Ad[0, 0], n_dm),
                    ([-1, 6, 7, 8, 9], [-2, -3, -4, -5, 6, 7, 8, 9]),
                )
                env_0 = np.reshape(env_0, (1, -1))
                local_basis = linalg.null_space(onp.array(env_0))
                if basis is None:
                    basis = local_basis
                else:
                    basis = linalg.block_diag(basis, local_basis)
    # basis = _filter_null_modes(tensors, basis)
    return basis


def filter_null_modes(tensors, basis):
    def _apply_ops_h(A, B, ops):
        for i in A.x_major():
            with cur_loc(i):
                op_r = ops[0, 0]
                op_l = ops[-1, 0]
                phi = cmath.exp(1j * sim_config.px)
                B[0, 0] = phi * ncon((A[0, 0], op_r), ([-1, 1, -3, -4, -5], [1, -2]))
                B[0, 0] = B[0, 0] - ncon(
                    (A[0, 0], op_l), ([-1, -2, -3, 1, -5], [-4, 1])
                )
        return B

    def _apply_ops_v(A, B, ops):
        for i in A.x_major():
            with cur_loc(i):
                op_d = ops[0, 0]
                op_u = ops[0, -1]
                phi = cmath.exp(-1j * sim_config.py)
                B[0, 0] = phi * ncon((A[0, 0], op_u), ([-1, -2, 1, -4, -5], [-3, 1]))
                B[0, 0] = B[0, 0] - ncon(
                    (A[0, 0], op_d), ([-1, -2, -3, -4, 1], [1, -5])
                )
        return B

    ops_h = TList(pattern=tensors.A.pattern)
    ops_v = TList(pattern=tensors.A.pattern)
    D = sim_config.D
    for i in tensors.A.x_major():
        with cur_loc(i):
            ops_h[0, 0] = np.zeros((D, D))
            ops_v[0, 0] = np.zeros((D, D))

    key = random.PRNGKey(0)
    nulls = None
    for i in range(sim_config.D**2 * len(tensors.A)):
        key, subkey = random.split(key)
        v = random.normal(key, (ops_h.tot_numel(),))
        ops_h = ops_h.fill(v)
        new_vec = _apply_ops_h(tensors.A, tensors.B, ops_h).pack_data()
        new_vec = np.expand_dims(new_vec, 1)
        if i == 0:
            nulls = new_vec
        else:
            nulls = np.hstack((nulls, new_vec))
            nulls = linalg.orth(nulls)
        v = random.normal(key, (ops_v.tot_numel(),))
        ops_v = ops_v.fill(v)
        new_vec = _apply_ops_v(tensors.A, tensors.B, ops_v).pack_data()
        new_vec = np.expand_dims(new_vec, 1)
        if i == 0:
            nulls = new_vec
        else:
            nulls = np.hstack((nulls, new_vec))

    nulls = basis.T.conjugate() @ nulls
    basis = basis @ linalg.null_space(nulls.conjugate().T)
    return basis


def get_dms(ts, only_gs=False):
    """Returns the two-site reduced density matrices

    This function relies on the Nested class, which contains
    tuples of different variants of site/boundary tensors.
    These variants contain either no B/Bd tensors, only a B
    tensor, only a Bd tensor or both a B and a Bd tensor.

    When the Nested tensors are contracted, all possible combinations
    that result again in one of these variants are computed and
    summed when there are multiple results in the same variant class.

    As a result, the different terms are summed on the fly during the
    contraction, which greatly reduces the computational cost.

    For example, the reduced density matrices contain 12*12=144 terms
    each (all possible locations of B and Bd tensors in the various
    boundaries), so that would make the energy evaluation 144 times
    as expensive as the ground state energy evaluation.
    Using this resummation, the total cost reduces to the maximal number
    of combinations in each contraction of pairs of tensors, 9, leading
    to a total computational cost of less than 9 times the ground state
    energy evaluation cost (the site tensors contain only two variants,
    so not every contraction contains 9 combinations).

    See the notes in nested.py for more details

    roh,rov are Nested tensors, with the following content:
        ro*[0]: ground state (no B/Bd tensors)
        ro*[1]: all terms with a single B tensor
        ro*[2]: all terms with a single Bd tensor
        ro*[3]: all terms with both a single B and Bd tensor

    The horizontal and vertical dms are located with respect
    to site (0,0) as follows:

    A_up (0,0)
     |
    A_mid (0,1) -- A_right (1,1)
    """

    if only_gs:
        A = ts.A
        Ad = ts.Ad
        C1 = ts.Cs(0)
        C2 = ts.Cs(1)
        C3 = ts.Cs(2)
        C4 = ts.Cs(3)
        T1 = ts.Ts(0)
        T2 = ts.Ts(1)
        T3 = ts.Ts(2)
        T4 = ts.Ts(3)
    else:
        # The 'all_*' functions return Nested tensors, so for example
        # ts.all_Cs(0) contains (C1, B_C1, Bd_C1, BB_C1)
        A = ts.all_A
        Ad = ts.all_Ad
        C1 = ts.all_Cs(0)
        C2 = ts.all_Cs(1)
        C3 = ts.all_Cs(2)
        C4 = ts.all_Cs(3)
        T1 = ts.all_Ts(0)
        T2 = ts.all_Ts(1)
        T3 = ts.all_Ts(2)
        T4 = ts.all_Ts(3)

    # Tensors that are part of the vertical reduced density matrix
    v_tensors = [
        C1[-1, -1],
        C2[1, -1],
        C3[1, 2],
        C4[-1, 2],
        T1[0, -1],
        T2[1, 0],
        T2[1, 1],
        T3[0, 2],
        T4[-1, 0],
        T4[-1, 1],
        A[0, 0],
        A[0, 1],
        Ad[0, 0],
        Ad[0, 1],
    ]

    # Tensors that are part of the horizontal reduced density matrix
    h_tensors = [
        C1[-1, 0],
        C2[2, 0],
        C3[2, 2],
        C4[-1, 2],
        T1[0, 0],
        T1[1, 0],
        T2[2, 1],
        T3[0, 2],
        T3[1, 2],
        T4[-1, 1],
        A[0, 1],
        A[1, 1],
        Ad[0, 1],
        Ad[1, 1],
    ]

    # Regular variant
    roh = _get_dm_h(*h_tensors)
    rov = _get_dm_v(*v_tensors)

    return roh, rov


def _get_dm_v(C1, C2, C3, C4, T1, T2u, T2d, T3, T4u, T4d, Au, Ad, Adu, Add):
    """Regular variant

    A_up (0,0)
     |
    A_mid (0,1)
    """
    py = sim_config.py

    # Upper half
    Cc1 = ncon([C1, T1, T4u, Au], "dm_up_Cc1")
    Cc2 = ncon([C2, T2u, Adu], "dm_up_Cc2")
    Cc2 = ncon([Cc1, Cc2], "dm_up")

    # Lower half
    Cc1 = ncon([C4.shift(py), T3.shift(py), T4d.shift(py), Ad.shift(py)], "dm_low_Cc1")
    Cc3 = ncon([C3.shift(py), T2d.shift(py), Add.shift(py)], "dm_low_Cc3")
    Cc3 = ncon([Cc1, Cc3], "dm_low")

    # Contract
    rov = ncon([Cc3, Cc2], "dm_rov")
    return rov


def _get_dm_h(C1, C2, C3, C4, T1l, T1r, T2, T3l, T3r, T4, Al, Ar, Adl, Adr):
    """Regular variant

    A_mid (0,1) -- A_right (1,1)
    """
    px = sim_config.px

    # Left half
    Cc1 = ncon([C4, T3l, T4, Al], "dm_low_Cc1")
    Cc2 = ncon([C1, T1l, Adl], "dm_left_Cc2")
    Cc2 = ncon([Cc1, Cc2], "dm_left")

    # Right half
    Cc1 = ncon([C2.shift(px), T1r.shift(px), Ar.shift(px)], "dm_right_Cc1")
    Cc3 = ncon(
        [C3.shift(px), T2.shift(px), T3r.shift(px), Adr.shift(px)], "dm_right_Cc3"
    )
    Cc3 = ncon([Cc1, Cc3], "dm_right")

    # Contract
    roh = ncon([Cc2, Cc3], "dm_roh")
    return roh
