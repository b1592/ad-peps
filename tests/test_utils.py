import unittest
import adpeps
import scipy.io as sio
import numpy as np

from adpeps.utils import tlist

# @unittest.skip("TList being worked on")
class TestTList(unittest.TestCase):

    def setUp(self):
        self.np_l = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
        self.l         = tlist.TList(self.np_l)

    def test_cur_loc(self):
        for i in range(self.l.size[0]):
            for j in range(self.l.size[1]):
                with tlist.cur_loc(i,j):
                    # Note that the tlist is indexed row-major
                    self.assertEqual(self.l[0,0], self.np_l[j,i])
                # Outside of context
                self.assertEqual(self.l[i,j], self.np_l[j,i])

    def test_hold_write(self):
        l = self.l.copy()
        orig_l0 = l[0,0]
        with tlist.hold_write(l):
            l[0,0] = 500
            self.assertEqual(l[0,0], orig_l0)
            self.assertTrue(l._hold_write)
        self.assertEqual(l[0,0], 500)
        self.assertTrue(l._tmpdata is None)
        self.assertFalse(l._hold_write)

    def test_pattern_cur_loc(self):
        base_l = [10,20,30]
        np_l = np.array([[10,20,30],[20,30,10]])
        l = tlist.TList(base_l, pattern=[[0,1,2],[1,2,0]])
        for i in range(l.size[0]):
            for j in range(l.size[1]):
                with tlist.cur_loc(i,j):
                    # Note that the tlist is indexed row-major
                    self.assertEqual(l[0,0], np_l[j,i])
                # Outside of context
                self.assertEqual(l[i,j], np_l[j,i])

    def test_pattern_ctx_cur_loc(self):
        base_l = [10,20,30]
        np_l = np.array([[10,20,30],[20,30,10]])
        with tlist.set_pattern([[0,1,2],[1,2,0]]):
            l = tlist.TList(base_l)
        l2 = tlist.TList(base_l, pattern=[[0,1,2],[1,2,0]])
        self.assertTrue(l == l2)
        for i in range(l.size[0]):
            for j in range(l.size[1]):
                with tlist.cur_loc(i,j):
                    # Note that the tlist is indexed row-major
                    self.assertEqual(l[0,0], np_l[j,i])
                # Outside of context
                self.assertEqual(l[i,j], np_l[j,i])

    def test_pattern_from_full_cur_loc(self):
        base_l = [10,20,30]
        np_l = np.array([[10,20,30],[20,30,10]])
        l = tlist.TList(base_l, pattern=[[0,1,2],[1,2,0]])
        l2 = tlist.TList(np_l, pattern=[[0,1,2],[1,2,0]])
        self.assertTrue(l == l2)
        for i in range(l.size[0]):
            for j in range(l.size[1]):
                with tlist.cur_loc(i,j):
                    # Note that the tlist is indexed row-major
                    self.assertEqual(l[0,0], np_l[j,i])
                # Outside of context
                self.assertEqual(l[i,j], np_l[j,i])

    def test_pattern_hold_write(self):
        base_l = [10,20,30]
        l = tlist.TList(base_l, pattern=[[0,1,2],[1,2,0]])
        orig_l0 = l[0,0]
        with tlist.hold_write(l):
            l[0,0] = 500
            self.assertEqual(l[0,0], orig_l0)
            self.assertTrue(l._hold_write)
        self.assertEqual(l[0,0], 500)
        self.assertTrue(l._tmpdata is None)
        self.assertFalse(l._hold_write)
