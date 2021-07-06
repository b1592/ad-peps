"""
Main runner
"""

import argparse
from .simulation import run_ipeps_gs, run_ipeps_exci

from jax.config import config
config.update("jax_enable_x64", True)

import adpeps
from adpeps.utils import io

"""
    Main executable module for iPEPS simulations
    
    Select one of two modes:
        > python3 -m adpeps gs <config_file>
        > python3 -m adpeps exci <config_file> -p <momentum_ix>

    Where <config_file> corresponds to a .yaml configuration file

    The following environment variables can be set for the locations 
    of configuration and output files:
        - CONFIGDIR: base folder where the simulations look for configuration 
                files
        - DATADIR: base folder where output data is saved

    The naming conventions for both modes are defined in utils.io as follows:
        - Ground states: <DATADIR>/<out_prefix>_<model>_D<D>_X<chi>.npz
        - Excited states: <DATADIR>/<out_prefix>_<model>_D<D>_X<chi>/<p_ix>_<px>_<py>.npz
"""

def get_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', dest='version',
                        action='store_true',
                        help='Show version')

    subparsers = parser.add_subparsers(help='Simulation mode', dest='sim_mode')

    # Ground-state parser
    parser_gs = subparsers.add_parser('gs', help='Ground-state simulation')
    parser_gs.add_argument('config_file', type=str,
                help='Configuration (.yml) file for the simulation options')

    # Excited-state parser
    parser_exci = subparsers.add_parser('exci', help='Excited-state simulation')
    parser_exci.add_argument('config_file', type=str,
                        help='config file of excited-state simulation')
    parser_exci.add_argument('-p', '--p_ix', dest='momentum_ix',
                        default=0, type=int,
                        help='momentum index')
    parser_exci.add_argument('-e', '--eval', dest='evaluate',
                        action='store_true',
                        help='Prepare excitation base')
    parser_exci.add_argument('-i', '--init', dest='init',
                        action='store_true',
                        help='Prepare excitation base')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    if args.version:
        print('Adpeps version:')
        print(adpeps.__version__)

    elif args.sim_mode == 'gs':
        print('Running ground-state sim')
        args.config_file = io.localize_config_file(args.config_file)
        run_ipeps_gs.run(args.config_file)

    elif args.sim_mode == 'exci':
        print('Running excited-state sim')
        print(args.config_file)
        args.config_file = io.localize_config_file(args.config_file)
        print(args.config_file)
        if args.evaluate:
            run_ipeps_exci.evaluate(args.config_file, args.momentum_ix-1)
        elif args.init:
            run_ipeps_exci.prepare(args.config_file)
        else:
            run_ipeps_exci.run(args.config_file, args.momentum_ix-1)
