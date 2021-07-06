"""
    List object with additional features, used for storing 
    the iPEPS tensors

    Items in the list can be accessed by either a linear index 
    or a (i,j) double index, where i and j will be automatically 
    taken modulo the unit cell size (i.e. i = i % n_x)

    Additionally, convenience functions that work on tensors can 
    be defined for the whole list, e.g. conj()
"""

import contextlib
import jax
import jax.numpy as np
import numpy as onp

from .empty_tensor import EmptyT


@contextlib.contextmanager
def cur_loc(*loc: int):
    """ Shift the locations of the tensors relative to a 
        new zero (loc) while in this context

        Args:
            loc: shifts (x,y)

        Example:
            >>> l = TList([[1,2], [3,4]])
            >>> l[0,0]
            1
            >>> with cur_loc(1,0):
            >>>     l[0,0]
            2
            >>>     l[0,1]
            4
            >>> l[0,0]
            1

        Note that this applies to ALL TList objects while 
        inside the context
    """
    pre_patched_value = getattr(TList, '_loc')
    setattr(TList, '_loc', loc)
    yield TList
    setattr(TList, '_loc', pre_patched_value)

@contextlib.contextmanager
def hold_write(*lists: 'TList'):
    """ Hold off on writing to the list while 
        inside the context

        Args:
            lists: one or more TList objects that should have the writing 
                action delayed until the context is disabled

        Example:
            >>> l = TList([[1,2], [3,4]])
            >>> with hold_write(l):
            >>>    l[0,0] = 100
            >>>    l[0,0] 
            1
            >>> l[0,0]
            100
    """
    for l in lists:
        l._hold_write = True
    yield
    for l in lists:
        l._purge_tmp()

@contextlib.contextmanager
def set_pattern(pattern):
    """ Set pattern for all new TLists that are created while 
        the context is active

        Args:
            pattern:
    """

    pre_patched_value = getattr(TList, '_default_pattern')
    setattr(TList, '_default_pattern', pattern)
    yield TList
    setattr(TList, '_default_pattern', pre_patched_value)

class TList:
    _loc             = (0,0)
    _default_pattern = None
    _changed         = None

    def __init__(self, data=None, shape=None, pattern=None, empty_obj=[[]]):
        self._tmpdata    = None
        self.pattern     = pattern
        self._hold_write = False
        self.empty_obj   = empty_obj
        if pattern is None and self._default_pattern is not None:
            self.pattern = self._default_pattern
        if self.pattern is None:
            if data is not None:
                try:
                    iter(data) # Check if iterable
                    data = np.array(data, dtype='object')
                    self._data = data.reshape([-1], order='C').tolist()
                    if data.ndim == 1:
                        self.size = (data.shape[0], 1)
                    else:
                        self.size = (data.shape[1], data.shape[0])
                except:
                    self._data = [data]
                    self.size = (1,1)
            elif shape is not None:
                self._data = (shape[0]*shape[1]) * empty_obj
                self.size = shape
            else:
                self._data = None
                self.size = ()
        else:
            self.pattern = np.array(self.pattern)
            self.size = (self.pattern.shape[1], self.pattern.shape[0])
            if data is not None:
                try:
                    iter(data) # Check if iterable
                    data = np.array(data, dtype='object')
                    if data.size == np.unique(self.pattern).size:
                        self._data = data.reshape([-1], order='C').tolist()
                    else:
                        self._data = np.unique(self.pattern).size * empty_obj
                        for j in range(self.pattern.shape[1]):
                            for i in range(self.pattern.shape[0]):
                                self._data[self.pattern[i,j]] = data[i,j]
                except:
                    self._data = [data]
                    self.size = (1,1)
            else:
                self._data = np.unique(self.pattern).size * empty_obj
                assert len(self._data) == np.unique(self.pattern).size, \
                        "Data must contain one element for each unique identifier in pattern"
        self.reset_changed()

    def x_major(self):
        return (self._conv_ix((x,y)) for y in range(self.size[1]) for x in range(self.size[0]))

    def y_major(self):
        return (self._conv_ix((x,y)) for x in range(self.size[0]) for y in range(self.size[1]))

    def __len__(self):
        return len(self._data)

    def mean(self):
        try:
            finite_elems = [x for x in self._data if isfinite(x)]
            return sum(finite_elems) / len(finite_elems)
        except Exception as e:
            return sum(self._data) / len(self)

    def sum(self):
        try:
            finite_elems = [x for x in self._data if isfinite(x)]
            return sum(finite_elems)
        except Exception as e:
            return sum(self._data)

    def normalize(self):
        new_list = TList(shape=self.size, pattern=self.pattern)
        new_list._data = [a / np.max(np.abs(a)) for a in self._data]
        return new_list

    def conj(self):
        new_list = TList(shape=self.size, pattern=self.pattern)
        new_list._data = [a.conj() for a in self._data]
        return new_list
    def items(self):
        return [a.item() for a in self._data]

    def pack_data(self):
        data = []
        for a in self._data:
            data.append(np.reshape(a, (-1,)))
        return np.concatenate(data)

    def reset_changed(self):
        if self._data is not None:
            self._changed = len(self._data) * [False]
        return self

    def mark_changed(self, linear_ix):
        if self._changed is not None:
            self._changed[linear_ix] = True

    def is_changed(self, *ix):
        if self._changed is None:
            return False
        linear_ix = self._conv_ix(ix)
        return self._changed[linear_ix]

    def fill(self, data, d=None, D=None):
        new_list = TList(shape=self.size, pattern=self.pattern)
        offset = 0
        new_data = []
        for i,a in enumerate(self):
            siz = a.size
            # new_data.append(np.reshape(data[offset:offset+siz], (d, D, D, D, D)))
            new_data.append(np.reshape(data[offset:offset+siz], a.shape))
            offset = offset + siz
        new_list._data = new_data
        return new_list

    def tot_numel(self):
        return sum([a.size for a in self._data])

    def stop_gradient(self):
        new_list = TList(shape=self.size, pattern=self.pattern)
        new_list._data = [jax.lax.stop_gradient(a) if len(a)>0 else a for a in self._data]
        return new_list

    def _conv_ix(self, ix):
        if isinstance(ix, (tuple,list)):
            if len(self._loc) == 1:
                # shift_i, shift_j = onp.unravel_index(self._loc[0], self.size, order='F')
                shift_j, shift_i = np.unravel_index(self._loc[0], self.size)
            else:
                shift_i, shift_j = self._loc
            i = (ix[0] + shift_i) % self.size[0]
            j = (ix[1] + shift_j) % self.size[1]
            # linear_ix = np.ravel_multi_index((i,j), self.size, order='F')
            linear_ix = self._linear_ix(i,j)
        else:
            linear_ix = ix
        return linear_ix

    def _linear_ix(self, i, j):
        if self.pattern is not None:
            return self.pattern[j][i]
        else:
            return np.ravel_multi_index((i,j), self.size, order='F')

    def _purge_tmp(self):
        self._tmpdata    = None
        self._hold_write = False

    def __eq__(self, other):
        if self._data != other._data:
            return False
        if self.pattern is not None:
            if other.pattern is None:
                return False
            if not (self.pattern == other.pattern).all():
                return False
        return True

    def __getitem__(self, ix):
        linear_ix = self._conv_ix(ix)
        if self._tmpdata is not None and self._tmpdata[linear_ix] is not None:
            return self._tmpdata[linear_ix]
        return self._data[linear_ix]

    def __setitem__(self, ix, value):
        linear_ix = self._conv_ix(ix)
        if self._hold_write:
            if self._tmpdata is None:
                self._tmpdata = [None] * len(self)
            self._tmpdata[linear_ix] = self._data[linear_ix]
        self._data[linear_ix] = value
        self.mark_changed(linear_ix)

    def __repr__(self):
        if self._data is None:
            return "TList{}[]"
        repr_str = "TList{"
        if self._loc is not None:
            repr_str += "Loc=" + self._loc.__repr__()
        if self.pattern is not None:
            repr_str += ",Pat=" + self.pattern.__repr__()
        repr_str += ",Size=" + self.size.__repr__()
        repr_str += "}["
        for j in range(self.size[1]):
            repr_str += "["
            for i in range(self.size[0]):
                try:
                    repr_str += f"{self[i,j].shape}"
                except:
                    repr_str += self[i,j].__repr__()
                if i < self.size[0]-1:
                    repr_str += ", "
            if j < self.size[1]-1:
                repr_str += "], "
            else:
                repr_str += "]]"
        return repr_str

    @staticmethod
    def empty_like(T, empty_obj=None):
        if empty_obj is None:
            empty_obj = T.empty_obj
        return TList(shape=T.size, pattern=T.pattern, empty_obj=empty_obj)

def isfinite(x):
    try:
        return len(x) > 0
    except Exception as e:
        return np.isfinite(np.array(x))
