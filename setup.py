# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import versioneer

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='adpeps',
    version=versioneer.get_version(),
    description='Basic AD iPEPS code for ground states and excitations',
    long_description=readme,
    author='Boris Ponsioen',
    author_email='b.g.t.ponsioen@uva.nl',
    url='https://github.com/b1592/ad-peps',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    install_requires=[
        'jax>=0.2.12',
        'jaxlib>=0.1.65',
        'pyyaml',
        'numpy',
        'scipy',
    ],
)
