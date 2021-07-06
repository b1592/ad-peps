"""
    iPEPS module for optimization with CTM

    For an example of how to run a simulation see :mod:`adpeps.simulation.run_ipeps_gs`

    The module is initialized from one of the specific 
    model files, which return the initial boundary and 
    site tensors

    The list of parameters is set to the elements of the 
    individual site tensors

    Conventions for indices:

        - Site tensors::

            A: [phys, right, top, left, bottom]

        - Boundary tensors::

            C1: [right, bottom]
            C2: [left,  bottom]
            C3: [top,   left]
            C4: [right, top]
            T1: [right, left, ket, bra]
            T2: [top,   bottom, ket, bra]
            T3: [right, left, ket, bra]
            T4: [top,   bottom, ket, bra]


    Order of boundary tensors::

        C1 - T1 - C2
        |    |    |
        T4 - A  - T2
        |    |    |
        C4 - T3 - C3
"""

from functools import partial
import copy

from jax import random
import jax
import jax.numpy as np

from .ctm import run_ctm
from adpeps.ipeps import evaluation
from adpeps.ipeps import models
from adpeps.tensor.contractions import ncon
from adpeps.utils.ctmtensors import CTMTensors
from adpeps.utils.printing import print
from adpeps.utils.tlist import set_pattern, cur_loc, TList
import adpeps.ipeps.config as sim_config

class iPEPS:
    """ Initialize the iPEPS based on the settings in
        :mod:`adpeps.ipeps.config`
    """
    reinit_env = False

    def __init__(self):
        # Load model Hamiltonian, observables
        model = getattr(models, sim_config.model)
        self.H, self.observables = model.setup()

        # Initialize tensors
        self.d = self.H.shape[0]
        A = init_A_tensor(self.d, sim_config.D, sim_config.pattern)
        A = A.normalize()
        Ad = A.conj()
        Cs, Ts = init_ctm_tensors(A, A)
        self.tensors = CTMTensors(A, Ad, Cs, Ts)
        self.base_Cs, self.base_Ts = None, None

        # Initialize conv object
        self.convergence = {}

        self.reinit_env = True

    def run(self, params: np.ndarray) -> np.ndarray:
        """ 
        Run the simulation

        Args:
            params: variational parameters

        Returns:
            energy of iPEPS
        """

        if params is not None:
            self.fill(params)

        if self.reinit_env:
            # Construct new boundary tensors and perform ctm iterations 
            # until convergence
            # Note: gradient tracking is disabled for this function, so 
            # only the ctm steps in the code after this line will be tracked
            print('Performing CTM pre-steps without tracking')
            self.converge_boundaries()

        # Perform the ctm routine to obtain updated boundary tensors
        print('Performing CTM')
        self.tensors, conv = run_ctm(self.tensors, sim_config.chi, conv_fun=self.compute_energy)

        # Evaluate energy
        res = self.evaluate()

        # Stop downstream gradient tracking for iPEPS tensors, 
        # so they become regular arrays that can be saved
        self.tensors.stop_gradient(only_boundaries=False)

        return res

    def compute_energy(self, tensors):
        E, _ = evaluation.get_gs_energy(self.H, tensors)
        return E

    def converge_boundaries(self):
        """ Performs CTM on the boundary tensors until convergence,
            without gradient tracking 
        """
        # Make a non-tracking version of the iPEPS tensors
        orig_A = copy.deepcopy(self.tensors.A)
        self.tensors.A = self.tensors.A.stop_gradient()
        self.tensors.Ad = self.tensors.Ad.stop_gradient()

        # Initialize new boundary tensors
        Cs, Ts = init_ctm_tensors(self.tensors.A, self.tensors.Ad)
        self.tensors = CTMTensors(self.tensors.A, self.tensors.Ad, Cs, Ts)

        # Perform CTM update steps on the boundary tensors
        conv_fun = self.compute_energy
        self.tensors.stop_gradient()
        self.tensors, conv = run_ctm(self.tensors, sim_config.chi, conv_fun=conv_fun)
        self.tensors.stop_gradient()

        self.save_boundary_tensors()

        # Restore the original (tracking) site tensors
        self.tensors.A = orig_A
        self.tensors.Ad = orig_A.conj()

    def save_boundary_tensors(self):
        self.base_Cs = copy.deepcopy(self.tensors.Cs)
        self.base_Ts = copy.deepcopy(self.tensors.Ts)

    def evaluate(self):
        E = self.compute_energy(self.tensors)
        print('Energy:', jax.lax.stop_gradient(E).item())
        return E


    """ Input/output methods """

    def numel(self):
        """ Number of variational parameters """
        return self.tensors.A.tot_numel()
    
    def parse_elements(self, elements):
        """ Returns site tensors filled with the input elements """
        assert elements.size == self.numel(), f"Size of input vector ({elements.size}) does not \
                match the number of parameters of the iPEPS ({self.numel()})"
        return self.tensors.A.fill(elements, self.d, sim_config.D)

    def fill(self, A):
        """ Fill the site tensors with the elements
            The elements can be specified either as a list of (d,D,D,D,D)-dimensional 
            arrays or one 1-dimensional array of all elements concatenated
        """
        if isinstance(A, np.ndarray) and A.ndim == 1:
            # Input is vector of elements
            A               = self.parse_elements(A)
            self.tensors.A  = A
            self.tensors.Ad = A.conj()
        else:
            # Input is a list of arrays
            assert len(A) == len(self.tensors.A), "Number of input tensors does not match \
                    the number of site tensors of the iPEPS"
            for i in range(len(self.tensors.A)):
                self.tensors.A._data[i] = A[i]
                self.tensors.Ad._data[i] = A[i].conj()


class iPEPS_exci(iPEPS):
    """ Excited-state variant of the iPEPS class
    """

    reinit_env = False

    def __init__(self):
        super().__init__()
        self.substract_gs_energy()

    def normalize_gs(self):
        nrm, nrm0, envBs, nrms0 = evaluation.compute_exci_norm(self.tensors)
        print(f"GS norm {nrms0[0]}", level=1)
        self.tensors.A._data    = [a/np.sqrt(np.abs(nrms0[i])) for i, a in enumerate(self.tensors.A)]
        self.tensors.Ad         = self.tensors.A.conj()
        nrm, nrm0, envBs, nrms0 = evaluation.compute_exci_norm(self.tensors)
        print(f"GS norm {nrm0}", level=1)

    def substract_gs_energy(self):
        E, _ = evaluation.get_gs_energy(self.H, self.tensors)
        E = E/2
        print(f"Substracting {E} from Hamiltonian", level=1)
        self.H = self.H - E * np.reshape(np.eye(self.H.shape[0]**2), self.H.shape)
        # self.H = np.reshape(np.eye(self.H.shape[0]**2), self.H.shape)

    def evaluate(self):
        E = evaluation.get_all_energy(self.H, self.tensors)
        nrm, _, envBs, _ = evaluation.compute_exci_norm(self.tensors)
        print('Energies:', jax.lax.stop_gradient(E[3]), jax.lax.stop_gradient(E[0]), level=0)
        print('Norm:', jax.lax.stop_gradient(nrm), level=0)
        return E[3], envBs

    def run_gc(self, *args):
        res, _ = self.run(*args)
        return res

    def compute_energy(self, tensors):
        E = evaluation.get_all_energy(self.H, tensors)
        nrm, *_ = evaluation.compute_exci_norm(tensors)
        print('Energies:', jax.lax.stop_gradient(E[3]), jax.lax.stop_gradient(E[0]), level=2)
        print('Norm:', jax.lax.stop_gradient(nrm), level=2)
        print('Normalized E:', jax.lax.stop_gradient(E[3])/jax.lax.stop_gradient(nrm), level=2)
        return E[3] / nrm

    def compute_orth_basis(self):
        return evaluation.get_orth_basis(self.tensors)


    """ Input/output methods """

    def fill(self, B):
        if isinstance(B, np.ndarray) and B.ndim == 1:
            # Input is vector of elements
            B               = self.parse_elements(B)
            self.tensors.B  = B
            self.tensors.Bd = B.conj()
        else:
            for i in range(len(self.tensors.A._data)):
                self.tensors.B._data[i] = B[i]
                self.tensors.Bd._data[i] = B[i].conj()


def init_A_tensor(d, D, pattern):
    """
        The elements will be randomized based on sim_config.seed
    """
    with set_pattern(pattern):
        A = TList()

    key = random.PRNGKey(sim_config.seed)
    for i in range(A.size[0]):
        for j in range(A.size[1]):
            with cur_loc(i,j):
                if not A.is_changed(0,0):
                    key, subkey = random.split(key)
                    A[0,0] = random.normal(key, (d,D,D,D,D))
    return A

def init_ctm_tensors(A, Ad):
    """
    Returns initital boundary T-tensors based on the unit cell tensors and 
    empty trivial boundary C-tensors
    """
    unit_cell = A.size
    D         = A[0].shape[1]

    Cs   = [TList(shape=unit_cell, pattern=A.pattern) for _ in range(4)]
    Ts   = [TList(shape=unit_cell, pattern=A.pattern) for _ in range(4)]

    for i in range(A.size[0]):
        for j in range(A.size[1]):
            with cur_loc(i,j):
                Cs[0][0,0] = np.expand_dims(np.array([1.]), axis=(1))
                Cs[1][0,0] = np.expand_dims(np.array([1.]), axis=(1))
                Cs[2][0,0] = np.expand_dims(np.array([1.]), axis=(1))
                Cs[3][0,0] = np.expand_dims(np.array([1.]), axis=(1))
                Ts[0][0,0] = np.expand_dims(ncon([A[0,0], Ad[0,0]], ([1,2,3,4,-1], [1,2,3,4,-2])), axis=(0,1))
                Ts[1][0,0] = np.expand_dims(ncon([A[0,0], Ad[0,0]], ([1,2,3,-1,4], [1,2,3,-2,4])), axis=(0,1))
                Ts[2][0,0] = np.expand_dims(ncon([A[0,0], Ad[0,0]], ([1,2,-1,3,4], [1,2,-2,3,4])), axis=(0,1))
                Ts[3][0,0] = np.expand_dims(ncon([A[0,0], Ad[0,0]], ([1,-1,2,3,4], [1,-2,2,3,4])), axis=(0,1))
    return Cs, Ts
