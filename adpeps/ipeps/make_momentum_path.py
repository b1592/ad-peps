from math import pi

import numpy as np


def make_momentum_path(name, with_plot_info=False):
    n_per_piece = 5
    plot_info = {}

    if name == "Bril1":
        kxs = np.concatenate(
            [
                lin_ex(pi, pi, 2 * n_per_piece),
                lin_ex(pi, pi / 2, n_per_piece),
                lin_ex(pi / 2, 0, n_per_piece),
                lin_ex(0, pi, 2 * n_per_piece),
                np.linspace(pi, pi / 2, n_per_piece),
            ]
        )
        kys = np.concatenate(
            [
                lin_ex(0, pi, 2 * n_per_piece),
                lin_ex(pi, pi / 2, n_per_piece),
                lin_ex(pi / 2, 0, n_per_piece),
                lin_ex(0, 0, 2 * n_per_piece),
                np.linspace(0, pi / 2, n_per_piece),
            ]
        )
        if with_plot_info:
            plot_info["xticks"] = {
                "ticks": [0, 9, 13, 17, 26, 30],
                "labels": [
                    "$M(\pi,0)$",
                    "$X(\pi,\pi)$",
                    "$S(\pi/2,\pi/2)$",
                    "$\Gamma(0,0)$",
                    "$M(\pi,0)$",
                    "$S(\pi/2,\pi/2)$",
                ],
            }
            return kxs, kys, plot_info
        return kxs, kys
    if name == "Bril1-negy":
        kxs = np.concatenate(
            [
                lin_ex(pi, pi, 2 * n_per_piece),
                lin_ex(pi, pi / 2, n_per_piece),
                lin_ex(pi / 2, 0, n_per_piece),
                lin_ex(0, pi, 2 * n_per_piece),
                np.linspace(pi, pi / 2, n_per_piece),
            ]
        )
        kys = -np.concatenate(
            [
                lin_ex(0, pi, 2 * n_per_piece),
                lin_ex(pi, pi / 2, n_per_piece),
                lin_ex(pi / 2, 0, n_per_piece),
                lin_ex(0, 0, 2 * n_per_piece),
                np.linspace(0, pi / 2, n_per_piece),
            ]
        )
        if with_plot_info:
            plot_info["xticks"] = {
                "ticks": [0, 9, 13, 17, 26, 30],
                "labels": [
                    "$M(\pi,0)$",
                    "$X(\pi,-\pi)$",
                    "$S(\pi/2,-\pi/2)$",
                    "$\Gamma(0,0)$",
                    "$M(\pi,0)$",
                    "$S(\pi/2,-\pi/2)$",
                ],
            }
            return kxs, kys, plot_info
        return kxs, kys
    elif name == "0-2pi":
        kxs = np.linspace(0, 2 * pi, 33)
        kys = np.linspace(0, 2 * pi, 33)
        return kxs, kys
    elif name == "0-2pi-negy":
        kxs = np.linspace(0, 2 * pi, 33)
        kys = -np.linspace(0, 2 * pi, 33)
        return kxs, kys
    elif name == "0-2pi-x":
        kxs = np.linspace(0, 2 * pi, 33)
        kys = np.linspace(0, 0, 33)
        if with_plot_info:
            plot_info["xticks"] = {
                "ticks": [0, 16, 32],
                "labels": ["$\Gamma(0,0)$", "$M(\pi,0)$", "$(2\pi,0)$"],
            }
            return kxs, kys, plot_info
        return kxs, kys
    if name == "Bril1xy":
        kys = np.concatenate(
            [
                lin_ex(pi, pi, 2 * n_per_piece),
                lin_ex(pi, pi / 2, n_per_piece),
                lin_ex(pi / 2, 0, n_per_piece),
                lin_ex(0, pi, 2 * n_per_piece),
                np.linspace(pi, pi / 2, n_per_piece),
            ]
        )
        kxs = np.concatenate(
            [
                lin_ex(0, pi, 2 * n_per_piece),
                lin_ex(pi, pi / 2, n_per_piece),
                lin_ex(pi / 2, 0, n_per_piece),
                lin_ex(0, 0, 2 * n_per_piece),
                np.linspace(0, pi / 2, n_per_piece),
            ]
        )
        if with_plot_info:
            plot_info["xticks"] = {
                "ticks": [0, 9, 13, 17, 26, 30],
                "labels": [
                    "$M2(0,\pi)$",
                    "$X(\pi,\pi)$",
                    "$S(\pi/2,\pi/2)$",
                    "$\Gamma(0,0)$",
                    "$M(0,\pi)$",
                    "$S(\pi/2,\pi/2)$",
                ],
            }
            return kxs, kys, plot_info
        return kxs, kys
    else:
        raise ValueError("Momentum path name not known")


def lin_ex(s, e, n):
    return np.linspace(s, e, n)[:-1]
