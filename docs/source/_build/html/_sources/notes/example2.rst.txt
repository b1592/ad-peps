.. _notes/example2:

Example: excitations
=========================================

.. note::
  This example continues from :ref:`the ground-state example<notes/example>` and requires an optimized ground state to start from.

Here we demonstrates how to use the :mod:`adpeps` package for computing excited states of the 2D Heisenberg model.

The configuration file `examples/heis_D2_exci.yaml` contains the following settings:

.. literalinclude:: ../../../examples/heis_D2_exci.yaml

Note that many of the options are the same as for the ground-state simulation, with the addition of the :attr:`adpeps.ipeps.config.momentum_path` setting, which controls which path through the Brillouin zone will be taken.

In the configuration for excited states you do not explicitly set the momentum, but choose a preset path of points in momentum space and pass the index for each simulation.

If we now start the simulation, we get the following response:

  >>> python -m adpeps exci 'heis_D2_exci' --p_ix=1
  ...
  Running excited-state sim
  ...
  Base file .../exci/heisenberg_D2_X40.base.npz not found. 
  Prepare the simulation first by running with option '-i'

What happened is that we first need to make some preparations for the simulation.
For excited-state simulations, we require the following:

  1. Well-converged CTM boundary tensors
  2. A basis for the excited-state tensors, orthogonal to the ground state

The preparations for the simulation need to be performed only once, resulting in a `base` simulation file that will be used by the simulations for every momentum.

  >>> python -m adpeps exci 'heis_D2_exci' -i
  ...
  Running excited-state sim
  ...
    | CTM step 1 conv: 1.036e-01 time: 3.64 obj: -0.665574
    | CTM step 2 conv: 3.137e-03 time: 4.23 obj: -0.662436
    | CTM step 3 conv: 7.887e-05 time: 4.0 obj: -0.662515
    | CTM step 4 conv: 1.388e-06 time: 2.2 obj: -0.662514
    | CTM step 5 conv: 2.455e-07 time: 1.35 obj: -0.662514
    | CTM step 6 conv: 3.044e-08 time: 1.35 obj: -0.662514
    | CTM step 7 conv: 4.673e-09 time: 1.37 obj: -0.662514
    | CTM step 8 conv: 4.467e-10 time: 1.36 obj: -0.662514
    | CTM step 9 conv: 5.029e-11 time: 1.35 obj: -0.662514
    | CTM step 10 conv: 5.612e-11 time: 1.38 obj: -0.662514
    | CTM step 11 conv: 2.801e-11 time: 1.36 obj: -0.662514
    | CTM step 12 conv: 1.204e-11 time: 1.65 obj: -0.662514
    | CTM step 13 conv: 4.936e-12 time: 1.42 obj: -0.662514
    | CTM step 14 conv: 1.989e-12 time: 1.39 obj: -0.662514
  GS norm 3.5890188873039093
  GS norm 1.0
  Substracting -0.33125703308289145 from Hamiltonian

Several steps have been performed: first a full CTM contraction of the ground-state network, followed by a normalization of the ground-state tensors.
Then the Hamiltonian is shifted by the ground-state energy expectation value, in order for the excitations to have energies relative to the ground state.
Finally, the basis is prepared and we have everything to get started.

  >>> python -m adpeps exci 'heis_D2_exci' --p_ix=0
  ...
  Running excited-state sim
  ...
  Starting simulation of basis vector 1/62
  Performing CTM
    | CTM step 1 conv: 4.236e+00 time: 3.19 obj: 5.173629
    | CTM step 2 conv: 3.539e-02 time: 1.38 obj: 5.138237
    | CTM step 3 conv: 6.139e-03 time: 1.27 obj: 5.132099
    | CTM step 4 conv: 2.116e-04 time: 1.24 obj: 5.132310
    | CTM step 5 conv: 6.148e-05 time: 1.28 obj: 5.132249
    | CTM step 6 conv: 3.704e-04 time: 1.25 obj: 5.131879
    | CTM step 7 conv: 2.493e-04 time: 1.25 obj: 5.131629
  Energies: 0.04418993415167889 1.5450126399606245e-10
  Norm: 0.008611083119254162
  ==========
  Finished basis vector 1/62
  -
  Starting simulation of basis vector 2/62
  Performing CTM
    | CTM step 1 conv: 2.004e+00 time: 0.99 obj: 3.335621
    | CTM step 2 conv: 6.150e-02 time: 1.26 obj: 3.397117
    | CTM step 3 conv: 2.017e-02 time: 1.28 obj: 3.376950
    | CTM step 4 conv: 3.838e-03 time: 1.26 obj: 3.380788
    | CTM step 5 conv: 3.650e-04 time: 1.3 obj: 3.380423
    | CTM step 6 conv: 1.246e-03 time: 1.29 obj: 3.381669
    | CTM step 7 conv: 1.288e-03 time: 1.31 obj: 3.380380
  Energies: 0.4285335373465171 1.5450126399606245e-10
  Norm: 0.12679967208649232
  ==========
  Finished basis vector 2/62
  ...

In this version of the algorithm, the full energy and norm overlap matrices will be computed.
Each of the basis vectors, as seen in the output above, is used as input in a separate CTM summation and the program will continue to run until all basis vectors have been used.
