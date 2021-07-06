""" Main ground-state executable script

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
from yaml import safe_load, dump
import jax
import jax.numpy as np
import numpy as onp

from adpeps.ipeps.ipeps import iPEPS, iPEPS_exci
from adpeps.utils import io
from adpeps.utils.printing import print
import adpeps.ipeps.config as sim_config

def run(config_file: str):
    """ Start the simulation

        Args:
            config_file: filename of the configuration file
    """

    energies   = []
    gradnorms  = []
    def verbose(xk, step_size=None):
        """ Output printing function """
        try:
            energies.append(obj.cached_out)
            gradnorms.append(obj.gradnrm)
        except:
            pass
        print(' ')
        print(' # ======================== #')
        print(' #      Step completed      #')
        print(' # ======================== #')
        print(' ')
        [print(' Step %3d  E: %.12f  |grad|: %2.8g' % (i,E,gradnorms[i])) for i,E in enumerate(energies)]
        print('\n')
        # np.savez(output_file, peps=peps, v=xk, gradnorms=gradnorms, energies=energies)

    print('Running')
    print(config_file)
    with open(config_file) as f:
        cfg = safe_load(f)

    # Show options
    print(dump(cfg))

    # Load the configuration file into the sim_config object
    sim_config.from_dict(cfg)

    # Initialize the iPEPS
    peps = iPEPS()

    output_file = io.get_gs_file()
    print(f"Output file {output_file}")

    if sim_config.resume and output_file.exists():
        loaded_sim = np.load(output_file, allow_pickle=True)
        peps       = loaded_sim['peps'].item()
        v          = loaded_sim['v']
        gradnorms  = list(loaded_sim['gradnorms'])
        energies   = list(loaded_sim['energies'])
        print("Resuming existing simulation")
        verbose(v)
    else:
        print("Starting new simulation")
        key = random.PRNGKey(sim_config.seed)
        v   = random.normal(key, (peps.numel(),))
        v   = v / np.max(np.abs(v))

    obj = Objective(peps)

    # Call SciPy's optimization function
    obj.return_gn = False
    xL = optimize.minimize(obj.out, v, method=sim_config.method, jac=obj.jac,
            callback=verbose, options={'gtol': 1e-6, 'disp': True,
                'maxiter':sim_config.max_iterations})

class Objective:
    """ Class containing the objective function for the optimizer
    """

    def __init__(self, peps: iPEPS):
        """
        Args:
            peps: iPEPS object to be optimized
        """
        self.cached_v   = None
        self.cached_out = None
        self.cached_jac = None

        self.peps       = peps
        """ iPEPS object """

        self.fun        = peps.run
        """ Objective function - CTM iterations until convergence 
            followed by the evaluation of the energy 
        """
        self.return_gn  = True

    @property
    def gradnrm(self):
        """ Norm of the gradient """
        return np.linalg.norm(self.cached_jac)

    def jac(self, v: np.ndarray) -> np.ndarray:
        """ Computes the vector-Jacobian product (gradient) of the
            iPEPS at the current point :attr:`v` in parameter space

            This function is designed to work with optimizers that make separate 
            calls to the objective function and the gradient, by caching both.

            Args:
                v: input variational parameters for the site tensors of 
                    the iPEPS

            Returns:
                gradient of the iPEPS with respect to the input parameters :attr:`v`
        """

        # Cast the regular numpy array into a Jax numpy array for gradient tracking
        v = np.array(v)
        v = v / np.max(np.abs(v))

        if (self.cached_jac is not None and self.cached_v is not None 
                and np.linalg.norm(v - self.cached_v) < 1e-14):
            return self.cached_jac

        # Call the forward + backward pass iPEPS code
        o, g = value_and_grad(self.fun)(v)

        # Convert to regular Numpy arrays so they can be passed to the optimizer 
        # and stored without any tracking information by Jax
        o = onp.array(o)
        v = onp.array(v)
        g = onp.array(g)

        # Cache the results
        self.cached_v   = v
        self.cached_out = o
        self.cached_jac = g

        if self.return_gn:
            return g, self.gradnrm
        else:
            return g

    def out(self, v):
        """ Computes the objective function (energy) of the iPEPS

            Since many optimizers make separate calls to the objective function 
            and the gradient, but usually require both results for any given 
            parameter vector v, this function calls the gradient as well.

            Args:
                v: input variational parameters for the site tensors of 
                    the iPEPS

            Returns:
                energy of the iPEPS at the point :attr:`v` in parameter space
        """
        if self.cached_v is not None and np.linalg.norm(v - self.cached_v) < 1e-14:
            return self.cached_out
        else:
            self.jac(v)
            return self.cached_out

    def check_grads(self, A=None):
        print('Checking gradient')
        self.peps.fill(A)
        self.peps.converge_boundaries()
        check_grads(self.peps.run, (A,), order=1, modes='rev')
        print('Done check')

