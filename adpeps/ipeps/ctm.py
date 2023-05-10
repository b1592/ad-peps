"""
    Main CTM code

    The individual site and boundary tensors come in a 
    special list-type object (TList), which has extra 
    indexing features, such as periodic boundary 
    conditions and shift contexts

    All ncon contractions are defined in contractions.yaml
"""

import time
from typing import Tuple

import jax
import jax.numpy as np

import adpeps.ipeps.config as sim_config
from adpeps.tensor.contractions import ncon
from adpeps.tensor.ops import diag_inv, svd
from adpeps.utils.ctmtensors import CTMTensors
from adpeps.utils.nested import Nested
from adpeps.utils.tlist import TList, cur_loc, set_pattern


def run_ctm(tensors, chi, conv_fun=None):
    ctm = CTM(tensors, chi, conv_fun)
    return ctm()


class CTM:
    """CTM class"""

    def __init__(self, tensors: CTMTensors, chi: int, conv_fun=None):
        """
        Args:
            tensors: input ctmtensors
            chi: boundary bond dimension
        """

        self.tensors = tensors
        self.chi = chi
        self.conv_fun = conv_fun
        self.tol = sim_config.ctm_conv_tol  # Convergence tolerance (singular values)
        self.min_iter = sim_config.ctm_min_iter
        self.max_iter = sim_config.ctm_max_iter

        self.singular_values = None
        self.last_convergence = None
        self.diffs = [None]
        self.n_steps = 0
        self.last_ctm_time = None
        self.convergence = np.nan
        self.condition_number = None

    def __call__(self):
        while not self.converged:
            self.show_progress()
            self.update()

        return self.tensors, self.convergence

    def update(self):
        """Perform an update of all boundary tensors"""
        start = time.time()
        self.tensors, s = renormalize(self.tensors, self.chi)
        end = time.time()

        try:
            s = jax.lax.stop_gradient(s)
            s_nz = s[s != 0] / np.max(s)
            cond_s = np.min(s_nz)
        except:
            cond_s = np.nan

        self.n_steps += 1
        self.singular_values = s
        self.last_ctm_time = round(end - start, 2)
        self.condition_number = cond_s

    def show_progress(self):
        """Print out the current progress"""
        if self.n_steps > 0 and sim_config.disp_level > 0:
            if self.conv_fun is not None:
                print(
                    f"  | CTM step {self.n_steps} conv: {self.diffs[-1]:.3e} time: {self.last_ctm_time} obj: {self.convergence:.6f}"
                )
            else:
                print(
                    f"  | CTM step {self.n_steps} conv: {self.diffs[-1]:.3e} time: {self.last_ctm_time}"
                )

    @property
    def converged(self):
        """Check convergence with supplied convergence function"""
        if self.conv_fun is not None:
            s = jax.lax.stop_gradient(self.conv_fun(self.tensors))
        else:
            s = self.singular_values

        self.last_convergence = self.convergence
        self.convergence = s
        try:
            diff = np.linalg.norm(self.convergence - self.last_convergence)
            self.diffs.append(diff)
        except:
            diff = np.nan

        if self.n_steps >= self.min_iter and self.diffs[-1] < self.tol:
            return True
        elif self.n_steps >= self.max_iter:
            return True
        else:
            return False


def renormalize(tensors: CTMTensors, chi: int) -> Tuple[CTMTensors, np.ndarray]:
    """
    Performs a CTM iteration
    Updates all sites in the unit cell

    Args:
        tensors: input ctmtensors
        chi: boundary bond dimension

    Returns:
        A tuple containing

        - **tensors** (*CTMTensors*): updated tensors
        - **S** (*np.ndarray*): singular values of C1 (for convergence)
    """

    with set_pattern(tensors.A.pattern):
        tensors, sl = do_left(tensors, chi)
        tensors = do_right(tensors, chi)
        tensors = do_top(tensors, chi)
        tensors = do_bottom(tensors, chi)

    # Singular values of C1[0,0] - to check for convergence
    S = sl[0]
    return (tensors, S)


""" 
    ---------------------
    Individual left, right, top and bottom moves
    ---------------------

    Each move consists of a loop through the sites of the unit cell 
    in which first the projectors are computed and then the boundary 
    tensors are updated.

    The loops are optimized for readability with a few tricks that are 
    implemented in the TList or CTMTensors classes:

    - cur_loc(x,y): with this context enabled, all TList objects have a 
        shift applied in their coordinates.
        For example:

        A[0,0] = 1
        A[1,0] = 2
        with cur_loc(1,0):
            print(A[0,0]) # => 2 (retrieves element A([0,0]+[1,0]) = A[1,0])

        Using this context, the operations in the inner loops can be written 
        without reference to the (i,j) loop indices, as if it's just written 
        for one site in the unit cell.

    - CTMTensors.hold(tensor1, ...): with this context enabled, any values 
        stored in tensor1 (and other designated tensors) are only put in a 
        temporary location, so that reading the tensor still yields the 
        original values. After the context exits, the values will be 
        overwritten by the temporary values.
        Example:

        # ts is a CTMTensors object containing site/boundary tensors
        ts.C1[0,0] = [1]
        ts.C2[0,0] = [2]
        with ts.hold('C1'):
            ts.C1[0,0] = [10]
            ts.C2[0,0] = [20]
            print(ts.C1[0,0]) # => [1] since the value [10] is not yet stored
            print(ts.C2[0,0]) # => [20]

        print(ts.C1[0,0]) # => [10] since the context has exited

        With this context, there is no need to store the updated boundary 
        tensors in the inner loops in temporary objects (since in CTM each 
        update step should be performed separately).

    - TList.is_changed(x,y): tracks whether any of the tensors in the TList 
        have been updated since the last call to TList.reset_changed().
        This is useful for unit cells with pattern restrictions, so that 
        boundary tensors that correspond to equivalent sites are only 
        computed once.

"""


def do_left(ts: CTMTensors, chi: int) -> Tuple[CTMTensors, np.ndarray]:
    """
    Perform left CTM move

    Args:
        ts: input tensors
        chi: boundary bond dimension

    Returns:
        A tuple containing

        - **tensors** (*CTMTensors*): updated tensors
        - **sl** (*np.ndarray*): singular values of C1 (for convergence)

    """
    A = ts.A
    unit_cell = A.size
    ts.C1.reset_changed()
    ts.C4.reset_changed()
    ts.T4.reset_changed()
    Pl = TList(shape=unit_cell)  # Upper projectors
    Plb = TList(shape=unit_cell)  # Lower projectors
    sl = TList(shape=unit_cell)  # Singular values (for convergence check)
    # Loop over x direction of the unit cell
    for i in range(A.size[0]):
        # Loop over y direction
        for j in range(A.size[1]):
            # Change the relative shift of the lists
            with cur_loc(i, j):
                if not Pl.is_changed(0, 1):
                    Pl[0, 1], Plb[0, 1], sl[0, 1] = get_projectors_left(ts, chi)
        # Only update the lists after the loop over j is completed
        with ts.hold("all_C1", "all_C4", "all_T4"):
            for j in range(A.size[1]):
                with cur_loc(i, j):
                    if not ts.C1.is_changed(0, 0):
                        ts.update(
                            ("C1", "C4", "T4"),
                            ([0, 0], [0, 0], [0, 0]),
                            renorm_left(ts, Pl, Plb),
                        )
    return ts, sl


def do_right(ts: CTMTensors, chi: int) -> CTMTensors:
    """
    Perform right CTM move

    Args:
        ts: input tensors
        chi: boundary bond dimension

    Returns:
        ts: updated tensors

    """
    A = ts.A
    unit_cell = A.size
    ts.C2.reset_changed()
    ts.C3.reset_changed()
    ts.T2.reset_changed()
    Pr = TList(shape=unit_cell)
    Prb = TList(shape=unit_cell)
    for i in range(A.size[0]):
        for j in range(A.size[1]):
            with cur_loc(i, j):
                if not Pr.is_changed(0, 1):
                    Pr[0, 1], Prb[0, 1], _ = get_projectors_right(ts, chi)
        with ts.hold("all_C2", "all_C3", "all_T2"):
            for j in range(A.size[1]):
                with cur_loc(i, j):
                    if not ts.C2.is_changed(1, 0):
                        ts.update(
                            ("C2", "C3", "T2"),
                            ([1, 0], [1, 0], [1, 0]),
                            renorm_right(ts, Pr, Prb),
                        )
    return ts


def do_top(ts: CTMTensors, chi: int) -> CTMTensors:
    """
    Perform top CTM move

    Args:
        ts: input tensors
        chi: boundary bond dimension

    Returns:
        ts: updated tensors

    """
    A = ts.A
    unit_cell = A.size
    ts.C1.reset_changed()
    ts.C2.reset_changed()
    ts.T1.reset_changed()
    Pt = TList(shape=unit_cell)
    Ptb = TList(shape=unit_cell)
    for j in range(A.size[1]):
        for i in range(A.size[0]):
            with cur_loc(i, j):
                if not Pt.is_changed(0, 0):
                    Pt[0, 0], Ptb[0, 0], _ = get_projectors_top(ts, chi)
        with ts.hold("all_C1", "all_C2", "all_T1"):
            for i in range(A.size[0]):
                with cur_loc(i, j):
                    if not ts.C1.is_changed(-1, 0):
                        ts.update(
                            ("C1", "C2", "T1"),
                            ([-1, 0], [2, 0], [0, 0]),
                            renorm_top(ts, Pt, Ptb),
                        )
    return ts


def do_bottom(ts: CTMTensors, chi: int) -> CTMTensors:
    """
    Perform bottom CTM move

    Args:
        ts: input tensors
        chi: boundary bond dimension

    Returns:
        ts: updated tensors

    """
    A = ts.A
    unit_cell = A.size
    ts.C3.reset_changed()
    ts.C4.reset_changed()
    ts.T3.reset_changed()
    Pb = TList(shape=unit_cell)
    Pbb = TList(shape=unit_cell)
    for j in range(A.size[1]):
        for i in range(A.size[0]):
            with cur_loc(i, j):
                if not Pb.is_changed(0, 0):
                    Pb[0, 0], Pbb[0, 0], _ = get_projectors_bottom(ts, chi)
        with ts.hold("all_C3", "all_C4", "all_T3"):
            for i in range(A.size[0]):
                with cur_loc(i, j):
                    if not ts.C3.is_changed(2, 1):
                        ts.update(
                            ("C3", "C4", "T3"),
                            ([2, 1], [-1, 1], [0, 1]),
                            renorm_bottom(ts, Pb, Pbb),
                        )
    return ts


""" 
    ---------------------
    Individual left, right, top and bottom projectors
    ---------------------

    The projectors are computed by contracting a corner of the 
    system (C-tensor + 2 T-tensors + A and Adagger tensors) 
    in the top (/left) half with a corner in the bottom (/right) 
    half and performing an svd
"""


def get_projectors_left(
    ts: CTMTensors, chi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the left projectors
    """
    tensors = (
        ts.C1[-1, -1],
        ts.C4[-1, 2],
        ts.T1[0, -1],
        ts.T3[0, 2],
        ts.T4[-1, 0],
        ts.T4[-1, 1],
        ts.A[0, 0],
        ts.Ad[0, 0],
        ts.A[0, 1],
        ts.Ad[0, 1],
    )
    return _get_projectors_left_impl(*tensors, chi)


def _get_projectors_left_impl(C1, C4, T1, T3, T4u, T4d, Au, Adu, Ad, Add, chi):
    Cs1 = ncon([C1, T1], "proj_left_Cs1")
    Q1 = ncon([Cs1, T4u, Au, Adu], "proj_left_Q1")

    Cs4 = ncon([C4, T3], "proj_left_Cs4")
    Q4 = ncon([Cs4, T4d, Ad, Add], "proj_left_Q4")

    Q4 = Q4.transpose([3, 4, 5, 0, 1, 2])
    return get_projectors(Q1, Q4, chi)


def get_projectors_right(
    ts: CTMTensors, chi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the right projectors
    """
    tensors = (
        ts.C2[2, -1],
        ts.C3[2, 2],
        ts.T1[1, -1],
        ts.T2[2, 0],
        ts.T2[2, 1],
        ts.T3[1, 2],
        ts.A[1, 0],
        ts.Ad[1, 0],
        ts.A[1, 1],
        ts.Ad[1, 1],
    )
    return _get_projectors_right_impl(*tensors, chi)


def _get_projectors_right_impl(C2, C3, T1, T2u, T2d, T3, Au, Adu, Ad, Add, chi):
    Cs2 = ncon([C2, T1], "proj_right_Cs2")
    Q2 = ncon([Cs2, T2u, Au, Adu], "proj_right_Q2")

    Cs3 = ncon([C3, T3], "proj_right_Cs3")
    Q3 = ncon([Cs3, T2d, Ad, Add], "proj_right_Q3")

    Q3 = Q3.transpose([3, 4, 5, 0, 1, 2])
    return get_projectors(Q2, Q3, chi)


def get_projectors_top(
    ts: CTMTensors, chi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the top projectors
    """
    tensors = (
        ts.C1[-1, -1],
        ts.C2[2, -1],
        ts.T1[0, -1],
        ts.T1[1, -1],
        ts.T2[2, 0],
        ts.T4[-1, 0],
        ts.A[0, 0],
        ts.Ad[0, 0],
        ts.A[1, 0],
        ts.Ad[1, 0],
    )
    return _get_projectors_top_impl(*tensors, chi)


def _get_projectors_top_impl(C1, C2, T1l, T1r, T2, T4, Al, Adl, Ar, Adr, chi):
    Cs1 = ncon([C1, T4], "proj_top_Cs1")
    Q1 = ncon([Cs1, T1l, Al, Adl], "proj_top_Q1")

    Cs2 = ncon([C2, T2], "proj_top_Cs2")
    Q2 = ncon([Cs2, T1r, Ar, Adr], "proj_top_Q2")

    Q2 = Q2.transpose([3, 4, 5, 0, 1, 2])
    return get_projectors(Q1, Q2, chi)


def get_projectors_bottom(
    ts: CTMTensors, chi: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the bottom projectors
    """
    tensors = (
        ts.C3[2, 2],
        ts.C4[-1, 2],
        ts.T2[2, 1],
        ts.T3[0, 2],
        ts.T3[1, 2],
        ts.T4[-1, 1],
        ts.A[0, 1],
        ts.Ad[0, 1],
        ts.A[1, 1],
        ts.Ad[1, 1],
    )
    return _get_projectors_bottom_impl(*tensors, chi)


def _get_projectors_bottom_impl(C3, C4, T2, T3l, T3r, T4, Al, Adl, Ar, Adr, chi):
    Cs4 = ncon([C4, T4], "proj_bottom_Cs4")
    Q4 = ncon([Cs4, T3l, Al, Adl], "proj_bottom_Q4")

    Cs3 = ncon([C3, T2], "proj_bottom_Cs3")
    Q3 = ncon([Cs3, T3r, Ar, Adr], "proj_bottom_Q3")

    Q3 = Q3.transpose([3, 4, 5, 0, 1, 2])
    return get_projectors(Q4, Q3, chi)


def get_projectors(T1: int, T2, chi):
    """Contracts the corners together and computes the
    projectors by performing an svd
    """
    full_chi = T1.shape[3] * T1.shape[4] * T1.shape[5]
    new_chi = min(full_chi, chi)

    Rho = ncon([T1, T2], ([-1, -2, -3, 1, 2, 3], [1, 2, 3, -4, -5, -6]))
    Rho_shape = Rho.shape
    Rho = np.reshape(Rho, [Rho_shape[0] * Rho_shape[1] * Rho_shape[2], -1])
    u, s, v = svd(Rho, new_chi, "n")
    u = np.reshape(u, [Rho_shape[0], Rho_shape[1], Rho_shape[2], -1])
    v = np.reshape(v.T, [Rho_shape[3], Rho_shape[4], Rho_shape[5], -1])
    inv_s = diag_inv(np.sqrt(s))

    P1 = ncon([T2, v, inv_s], "proj_P1")
    P2 = ncon([T1, u, inv_s], "proj_P2")

    P1 = P1.transpose([3, 0, 1, 2])
    P2 = P2.transpose([3, 0, 1, 2])
    return P1, P2, s


""" 
    ---------------------
    Individual left, right, top and bottom boundary tensor updates
    ---------------------

    The boundary tensors are updated for one site at a time from the 
    tensors of the previous iteration with the site tensors and truncated 
    by using the projectors

    In these functions, the boundary tensors can be wrapped as Nested tensors, 
    containing both ground-state and excited-state tensors.
    When using these Nested tensors, all different combinations are computed 
    automatically.
    For example:

        Nested({C1,B_C1,Bd_C1,BB_C1}) * Nested({T1,B_T1,Bd_T1,BB_T1}) ->
            Nested({
                (C1 * T1),
                (B_C1 * T1 + C1 * B_T1),
                (Bd_C1 * T1 + C1 * Bd_T1),
                (BB_C1 * T1 + B_C1 * Bd_T1 + Bd_C1 * B_T1 + C1 * BB_T1)
            })

    The phase shifts are only applied to the B and Bd parts of the Nested tensors
"""


def renorm_left(
    ts: CTMTensors, Pl: np.ndarray, Plb: np.ndarray
) -> Tuple[Nested, Nested, Nested]:
    """ """
    new_T4 = ncon(
        [Plb[0, 0], ts.all_T4[-1, 0], ts.all_A[0, 0], ts.all_Ad[0, 0], Pl[0, 1]],
        "doleft_T4",
        normalize=True,
    ).shift(-sim_config.px)

    Cs1 = ncon([ts.all_C1[-1, 0], ts.all_T1[0, 0]], "doleft_Cs1")
    new_C1 = ncon([Cs1, Pl[0, 1]], "doleft_C1", normalize=True).shift(-sim_config.px)

    Cs4 = ncon([ts.all_C4[-1, 0], ts.all_T3[0, 0]], "doleft_Cs4")
    new_C4 = ncon([Cs4, Plb[0, 0]], "doleft_C4", normalize=True).shift(-sim_config.px)

    return new_C1, new_C4, new_T4


def renorm_right(ts, Pr, Prb):
    new_T2 = ncon(
        [Prb[0, 0], ts.all_T2[2, 0], ts.all_A[1, 0], ts.all_Ad[1, 0], Pr[0, 1]],
        "doright_T2",
        normalize=True,
    ).shift(sim_config.px)

    Cs2 = ncon([ts.all_C2[2, 0], ts.all_T1[1, 0]], "doright_Cs2")
    new_C2 = ncon([Cs2, Pr[0, 1]], "doright_C2", normalize=True).shift(sim_config.px)

    Cs3 = ncon([ts.all_C3[2, 0], ts.all_T3[1, 0]], "doright_Cs3")
    new_C3 = ncon([Cs3, Prb[0, 0]], "doright_C3", normalize=True).shift(sim_config.px)

    return new_C2, new_C3, new_T2


def renorm_top(ts, Pt, Ptb):
    new_T1 = ncon(
        [Ptb[-1, 0], ts.all_T1[0, -1], ts.all_A[0, 0], ts.all_Ad[0, 0], Pt[0, 0]],
        "dotop_T1",
        normalize=True,
    ).shift(-sim_config.py)

    Cs1 = ncon([ts.all_C1[-1, -1], ts.all_T4[-1, 0]], "dotop_Cs1")
    new_C1 = ncon([Cs1, Pt[-1, 0]], "dotop_C1", normalize=True).shift(-sim_config.py)

    Cs2 = ncon([ts.all_C2[2, -1], ts.all_T2[2, 0]], "dotop_Cs2")
    new_C2 = ncon([Cs2, Ptb[1, 0]], "dotop_C2", normalize=True).shift(-sim_config.py)

    return new_C1, new_C2, new_T1


def renorm_bottom(ts, Pb, Pbb):
    new_T3 = ncon(
        [Pbb[-1, 0], ts.all_T3[0, 2], ts.all_A[0, 1], ts.all_Ad[0, 1], Pb[0, 0]],
        "dobottom_T3",
        normalize=True,
    ).shift(sim_config.py)

    Cs3 = ncon([ts.all_C3[2, 2], ts.all_T2[2, 1]], "dobottom_Cs3")
    new_C3 = ncon([Cs3, Pbb[1, 0]], "dobottom_C3", normalize=True).shift(sim_config.py)

    Cs4 = ncon([ts.all_C4[-1, 2], ts.all_T4[-1, 1]], "dobottom_Cs4")
    new_C4 = ncon([Cs4, Pb[-1, 0]], "dobottom_C4", normalize=True).shift(sim_config.py)

    return new_C3, new_C4, new_T3
