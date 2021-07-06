""" Contains utility class that represents a 'zero' (empty) tensor object """

import jax.numpy as np

class EmptyT:
    """ Empty tensor utility class, which can be used to represent a  
        'zero' tensor.

        The operations involving this type of tensor will return the expected 
        results, such as (Tensor * EmptyT -> EmptyT), removing the need for 
        checking if a tensor is empty in the part of the code where the 
        operation is called.
    """
    tag = None

    def __repr__(self):
        return "<empty>"

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return other

    def __rsub__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __neg__(self):
        return self

    def __rmul__(self, other):
        if isinstance(other, int):
            return [self for _ in range(other)]
        else:
            return self

    def __truediv__(self, other):
        return self

    def copy(self):
        return self

    def to_real(self):
        return self

    @property
    def real(self):
        return self

    @property
    def data(self):
        return []

    def transpose(self, *args):
        return self

    def __rtruediv__(self, other):
        return self

    def __len__(self):
        return 0

    def __array__(self):
        return np.array([])

    def item(self):
        return self

    def __getitem__(self, ix):
        return self

    def mult(self, other, *args):
        return self

    def to_complex(self):
        return self

    def complex(self):
        return False

    def is_finite(self):
        return True
