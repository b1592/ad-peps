# AD-PEPS

Basic implementation of iPEPS ground-state and excited-state optimization using automatic differentiation.

The package contains three main parts:

- The core iPEPS and CTM code, contained in `adpeps/ipeps`
- Executable scripts that run the simulations in `adpeps/simulation`
- Helper classes and functions, with custom contraction and other operations in `adpeps/tensor` and general utilities in `adpeps/utils`

## Installation

Download this repository

```
git clone https://github.com/b1592/ad-peps.git
```

Install the package using `pip`

```
cd ad-peps
pip install -e .
```

## Usage

As a general starting point for simulations the package can be executed as

```
python -m adpeps <options>
```

### Ground states

For ground states, the package can be used with the `gs` option:

```
python -m adpeps gs <config_file>
```

For each simulation, a configuration file in `yaml` format should be supplied that contains all relevant settings.
An example can be found in `examples/heis_D2.yaml`, with a description of each setting.

### Excited states

For excitations, the option to be used is `exci`:

```
python -m adpeps exci <config_file> <arguments>
```

In order to prepare for a simulation, first a ground-state simulation should be performed.
After this, a base file for the excited-state simulation can be created with the `-i` argument:

```
python -m adpeps exci <config_file> -i
```

This will converge the ground-state boundary tensors, normalize the ground state tensors, shift the Hamiltonian by the ground-state energy and finally compute a basis of vectors that are orthogonal to the ground state.

Once this is completed, the full simulation can be performed, which computes the full effective energy and norm matrices, by supplying the 'momentum index'.
This index corresponds to a certain path through momentum space, controlled by the `momentum_path` option in the configuration file.
For example, the first point in the `Bril1`, which is at `(pi,0)`, can be computed by running

```
python -m adpeps exci <config_file> -p 1
```

After the simulations are done, the results can be evaluated with the `-e` argument:

```
python -m adpeps exci <config_file> -e
```
