Installation
===================================

The quickest way of installing the :code:`adpeps` package is to clone the repository

.. code-block:: bash

  git clone <repo>


Method 1 (recommended): `conda`
------------------------------------------

The repository comes with an included :code:`environment.yml` file, which automatically installs a Python environment with all required packages, which can be used as follows

.. code-block:: bash

  cd ad-peps
  conda env create -f environment.yml
  conda activate adpeps

When the installation finishes, you can check that the package is working

.. code-block:: bash

  python -m adpeps -v

Method 2: `pip`
------------------------------------------

The package can also be installed via `pip`:

.. code-block:: bash

  cd ad-peps
  pip install -e .
