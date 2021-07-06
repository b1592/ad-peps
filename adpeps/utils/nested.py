""" Contains utility class that represents a collection of tensors of 
    different types, with operations that can be applied to all 
    contained tensors at once
"""

import cmath
import jax.numpy as np

from adpeps.types import TensorType


class Nested:
    """ This is a helper class for the efficient contraction of variants of tensors,
        used in the energy evaluation of excited states

        A Nested tensor contains the following variants (some may be empty):

            - :attr:`tensors[0]`: regular tensor (no B or Bd)
            - :attr:`tensors[1]`: (terms with) a single B tensor
            - :attr:`tensors[2]`: (terms with) a single Bd tensor
            - :attr:`tensors[3]`: (terms with) both a B and a Bd tensor
        
        When two Nested tensors x,y are contracted, all combinations are taken into account
        and the result is again a Nested tensor, filled with the following variants:

            - :attr:`tensors[0]: x[0] * y[0]`
            - :attr:`tensors[1]: x[1] * y[0] + x[0] * y[1]`
            - :attr:`tensors[2]: x[2] * y[0] + x[0] * y[2]`
            - :attr:`tensors[3]: x[3] * y[0] + x[2] * y[1] + x[1] * y[2] + x[0] * y[3]`

        By using Nested tensors in a (large) contraction, the many different terms are 
        resummed on the fly, leading to a potentially reduced computational cost

        Note:
            Most implented functions act as wrappers for the corresponding `numpy` functions 
            on the individual tensors
    """

    def __init__(self, tensors):
        self.tensors = tensors

    def normalize(self):
        """ Normalize the contained tensors by the largest value 
            of the first element of :attr:`self.tensors`
        """
        factor = np.abs(self[0]).max()
        return self * (1 / factor), factor

    def mult(self, other: TensorType, *args) -> 'Nested':
        """
            Args:
                other: other tensor-like object to contract with
                *args: arguments to be passed to the contraction method
                    (:code:`np.tensordot`)

            Returns:
                res: result of the contraction
        """
        def _mult_function(A, B, *args):
            if hasattr(A, 'mult'):
                return A.mult(B, *args)
            elif len(B) == 0:
                return B.mult(A, *args)
            return np.tensordot(A, B, *args)

        if isinstance(other, np.ndarray):
            new_data = 4 * [[]]
            new_data[0] = _mult_function(self.tensors[0], other, *args)
            new_data[1] = _mult_function(self.tensors[1], other, *args)
            new_data[2] = _mult_function(self.tensors[2], other, *args)
            new_data[3] = _mult_function(self.tensors[3], other, *args)
            return Nested(new_data)
        new_data = 4 * [[]]
        new_data[0] = _mult_function(self.tensors[0], other.tensors[0], *args)
        new_data[1] = _mult_function(self.tensors[1], other.tensors[0], *args) +\
                      _mult_function(self.tensors[0], other.tensors[1], *args)
        new_data[2] = _mult_function(self.tensors[2], other.tensors[0], *args) +\
                      _mult_function(self.tensors[0], other.tensors[2], *args)

        new_data[3] = _mult_function(self.tensors[3], other.tensors[0], *args) +\
                      _mult_function(self.tensors[2], other.tensors[1], *args) +\
                      _mult_function(self.tensors[1], other.tensors[2], *args) +\
                      _mult_function(self.tensors[0], other.tensors[3], *args)
        res = Nested(new_data)
        return res

    def transpose(self, *args) -> 'Nested':
        """ Applies :code:`transpose` to each contained tensor """
        new_data = [self.tensors[i].transpose(*args) for i in range(4)]
        return Nested(new_data)

    def __mul__(self, other):
        new_data = [self.tensors[i] * other for i in range(4)]
        return Nested(new_data)

    def __rmul__(self, other):
        new_data = [other * self.tensors[i] for i in range(4)]
        return Nested(new_data)

    def __truediv__(self, other):
        new_data = [self.tensors[i] / other for i in range(4)]
        return Nested(new_data)

    def __add__(self, other):
        if isinstance(other, Nested):
            new_data = [self.tensors[i] + other.tensors[i] for i in range(4)]
        else:
            new_data = [self.tensors[i] + other for i in range(4)]
        return Nested(new_data)

    def __radd__(self, other):
        return self + other

    def __getitem__(self, ix):
        return self.tensors[ix]

    def __setitem__(self, ix, value):
        self.tensors[ix] = value

    def __repr__(self):
        return "(Nested) " + self.tensors.__repr__()

    def __neg__(self):
        return Nested([-self.tensors[i] for i in range(4)])

    def shift(self, phi):
        new_data = [self.tensors[0],  self.tensors[1] * exp(phi), 
                 self.tensors[2] * exp(-phi), self.tensors[3]]
        return Nested(new_data)

    def __len__(self):
        try:
            return len(self.tensors[0])
        except Exception:
            return self.tensors[0].size

    @property
    def real(self):
        res = Nested([self.tensors[i].real for i in range(4)])
        return res

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def dims(self):
        return self.tensors[0].dims

    def check_contr_inds(self, other, *args, **kwargs):
        return self[0].check_contr_inds(other[0], *args, **kwargs)

    def numel(self):
        return self[0].numel()
    
    @classmethod
    def only_gs(cls, tensor, empty_obj=[]):
        return cls([tensor, empty_obj, empty_obj, empty_obj])

def exp(phi):
    return cmath.exp(1j * phi)
