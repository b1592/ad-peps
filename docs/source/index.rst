.. AD-PEPS documentation master file, created by
   sphinx-quickstart on Mon Apr 12 16:13:24 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AD-PEPS documentation
===================================

The AD-PEPS Python package is intended as a example of an implementation of iPEPS ground-state and excited-state algorithms using Automatic Differentiation, as described in <paper>
As such, the code is meant to illustrate the workings of the algorithms described the paper.
For real applications, this code would likely need to be further adapted and optimized, however the package is a complete implementation and can directly be used for simple calculations.

.. toctree::
   
   notes/install
   notes/start
   notes/example
   notes/example2


Reference
--------------

.. .. toctree::
..    :maxdepth: 2
   
..    ipeps

.. currentmodule:: adpeps

.. autosummary::
  :toctree: generated
  :recursive:

  ipeps.ipeps
  ipeps.config
  ipeps.ctm
  ipeps.models
  simulation
  utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
