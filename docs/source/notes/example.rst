.. _notes/example:

Example: ground state
===========================================

The package includes an example configuration for a ground-state simulation of the 2D Heisenberg model, defined by the Hamiltonian

.. math::

  H = J \sum_i S_i \cdot S_{i+1}~.

The configuration file `examples/heis_D2.yaml` contains the following:

.. literalinclude:: ../../../examples/heis_D2.yaml

This configures a simulation with bond dimension :code:`D=2` and boundary bond dimension :code:`chi=40`, using the model defined in :mod:`adpeps.ipeps.models.heisenberg`.

Now the simulation can be started by calling the :code:`adpeps` module with the name of this configuration file:

  >>> python -m adpeps gs 'heis_D2'
  WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
  Namespace(config_file='heis_D2', sim_mode='gs', version=False)
  Running ground-state sim
  ...

The simulation will continue to run and you should see output similar to this:

.. code-block::

  ...
  Performing CTM pre-steps without tracking
    | CTM step 1 conv: 4.935e-03 time: 3.17 obj: -0.658758
    | CTM step 2 conv: 7.918e-04 time: 3.74 obj: -0.659550
    | CTM step 3 conv: 1.234e-05 time: 6.42 obj: -0.659562
  Performing CTM
    | CTM step 1 conv: 3.171e-07 time: 8.8 obj: -0.659563
    | CTM step 2 conv: 2.108e-08 time: 4.07 obj: -0.659563
    | CTM step 3 conv: 8.173e-09 time: 2.93 obj: -0.659563
  Energy: -0.6595625579862193
  ...

The first cycle of iterations are not taken into account in the gradient computation, but make sure that the CTM iterations with gradient tracking start from some reasonably converged boundary tensors in order to avoid instabilities with initial CTM steps.

.. note::
  The convergence rate of the CTM depends on the variational parameters of the iPEPS and the settings of the simulation.
  Generally the convergence improves as the simulation approaches the optimum.

Whenever a step in the optimization has completed (this could take more than one cycle of CTM iterations depending on the type of optimizer), the module will output a summary of the steps so far:

.. code-block::

  ...
  # ======================== #
  #      Step completed      #
  # ======================== #

  Step   0  E: -0.376468389894  |grad|: 1.2103482
  Step   1  E: -0.505252956403  |grad|: 0.19064889
  Step   2  E: -0.517432085607  |grad|: 0.10910666
  Step   3  E: -0.578045570568  |grad|: 0.081472534
  Step   4  E: -0.589074339197  |grad|: 0.089438567
  Step   5  E: -0.597590746400  |grad|: 0.15282526
  Step   6  E: -0.612205652457  |grad|: 0.076385807
  Step   7  E: -0.628079118387  |grad|: 0.0684857
  Step   8  E: -0.642200026835  |grad|: 0.097849544
  Step   9  E: -0.649553574703  |grad|: 0.066648727
  Step  10  E: -0.653909263824  |grad|: 0.0264237
  Step  11  E: -0.655389076620  |grad|: 0.016836624
  Step  12  E: -0.656585389308  |grad|: 0.016954703
  Step  13  E: -0.657797020335  |grad|: 0.020011479
  Step  14  E: -0.658174755217  |grad|: 0.033691114
  Step  15  E: -0.659083649568  |grad|: 0.012202327
  Step  16  E: -0.659365377610  |grad|: 0.0064214407
  Step  17  E: -0.659562557986  |grad|: 0.007503111
  ...

The simulation will continue until :attr:`adpeps.ipeps.config.max_iterations` has been reached.
At any point the simulation can be stopped and continued later by restarting the module.

.. note::
  In case you would like the simulation to continue from an earlier saved simulation with the same configuration file, make sure to set :attr:`adpeps.ipeps.config.resume` :code:`= True`
