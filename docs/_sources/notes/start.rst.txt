.. _notes/start:

Getting Started
===================================

General
--------------

The main starting point for running simulations with the `adpeps` package is by loading the module directly via :code:`python -m adpeps`.

For both ground-state and excited-state simulations the configuration can be set via configuration :code:`.yaml` files.
Each option in the configuration file corresponds to an attribute of the :mod:`adpeps.ipeps.config` module.

The first argument for the module is the simulation mode (ground-state or excited-state):

.. code-block:: bash

  python -m adpeps {gs,exci} ...

.. note::
    The input configuration file location can be set via the 
    :envvar:`CONFIGDIR` variable. If it is not set, the default 
    input folder will be the `examples` subfolder of the package 
    root directory

.. note::
    The output data location can be set via the :envvar:`DATADIR` 
    variable. If it is not set, the default output folder will be 
    in the `simulations` subfolder of the package root directory.


Ground states
--------------

For ground-state simulations, the only required argument is the configuration file.

.. argparse::
  :ref: adpeps.__main__.get_parser
  :prog: python -m adpeps
  :path: gs


Excited states
--------------

For excited-state simulations, the first argument is again the name of a configuration file (note that the relevant options are different for excited-state simulations), and furthermore the `momentum index` :code:`-p` is required.

The momentum index refers to a point in momentum space :math:`(k_x, k_y)` defined in a specific path through the Brillouin zone.
The corresponding momentum path can be set via the :attr:`adpeps.ipeps.config.momentum_path` option.
By default, the `'Bril1'` path is taken, which follows the cut along high symmetry points :math:`(\pi,0) - (\pi,\pi) - (\pi/2,\pi/2) - (0,0) - (\pi,0) - (\pi/2,\pi/2)`

.. argparse::
  :ref: adpeps.__main__.get_parser
  :prog: python -m adpeps
  :nodefault:
  :path: exci

