#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import curve

class Parameter:
    """Log-aesthetic curve parameters.

    It contains the seven parameters necessary to identify a
    continuous Log-aesthetic curve and extra parameters for its
    discretization.

    NOTE: Parameters are computed for case=0.
    """

    def __init__(self):
        # Identities that must hold are:
        # For the un-scaled total length,
        #     l = (s1 - s0) = L / S = (N - 1) * h / S
        # For the segment length,
        #     h = L / (N - 1)
        # For the un-scaled segment length (called 'z' in the manuscript),
        #     ds = l / (N - 1) = h / S

        self.L = None
        self.case = None

        self.alpha = None
        self.s0 = None
        self.l = None
        self.S = None
        self.phi = None
        self.x0 = None
        self.y0 = None

        self.N = None
        self.h = None
        self.ds = None

    def get_ipopt_style(self):
        return np.array([self.alpha,
                         self.s0,
                         self.ds,
                         self.h,
                         self.phi,
                         self.x0,
                         self.y0])

    def set_ipopt_style(self, p, N, case):
        # IPOPT style:
        #     [0] -> alpha,
        #     [1] -> s0,
        #     [2] -> ds,
        #     [3] -> h,
        #     [4] -> phi,
        #     [5] -> x0,
        #     [6] -> y0

        self.alpha = p[0]
        self.s0 = p[1]
        self.S = p[3] / p[2]
        self.phi = p[4]
        self.x0 = p[5]
        self.y0 = p[6]
        self.case = case
        self.N = N
        self.h = p[3]
        self.ds = p[2]
        self.l = (N - 1) * p[2]
        self.L = (N - 1) * p[3]

    def printall(self):
        """Print on screen all the parameters.
        """

        for k, v in vars(self).items():
            if v is None:
                print(f'Warning: {k} has no value, ', end='')
            if k == 'ds':
                if self.ds is None:
                    print('(added for ds = l / (N - 1))')
                    self.ds = self.l / (self.N - 1)

        for k, v in vars(self).items():
            if v is not None:
                print(f'{k}\t{v:+0.3f}')
            else:
                print(f'{k}\tNone')
        pass

    def get_curve(self, *, case0=False):
        [x_lac, y_lac] = blac(alpha=self.alpha,
                              s0=self.s0,
                              ds=self.ds,
                              N=self.N)
        curve.simtransf(x_lac, y_lac,
                        S=self.S,
                        phi=self.phi,
                        x0=self.x0,
                        y0=self.y0)
        if not case0:
            curve.casetransf(x_lac, y_lac, self.case)
        return [x_lac, y_lac]

    def save_curve(self, file_name, **kwargs):
        [x, y] = self.get_curve()
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot(x, y, **kwargs)
        fig.savefig(file_name)
        plt.close(fig)


def blac(alpha=2, s0=0, ds=0.001, N=1000, s1=None, x0=0, y0=0):
    """Basic LAC."""

    # NOTE: abs() is used for pathological cases close to the edges of the
    # domain. In a general setting, we expect the argument to be positive.
    if alpha == 0:
        def ang(s):
            return 1 - np.exp(-s)
    elif alpha == 1:
        def ang(s):
            return np.log(np.abs(s + 1))
        #                 ^^^^^^
    else:
        def ang(s):
            return (np.abs(1 + alpha * s) ** (1 - 1/alpha) - 1) / (alpha - 1)
        #           ^^^^^^

    x = np.array([x0])
    y = np.array([y0])
    if s1 is not None:
        ds = (s1 - s0) / (N - 1)

    s = s0
    for n in range(N - 1):
        if (1 + alpha * s) < 0 and alpha > 0:
            raise ValueError("Error: domain of 's' is out of boundaries.")
        x = np.append(x, x[n] + np.cos(ang(s)) * ds)
        y = np.append(y, y[n] + np.sin(ang(s)) * ds)
        s += ds
    return np.array([x, y])

def param_internal(kappa, h, L, set_alpha=None, filename=None):
    R = -np.log(kappa)
    dR = -np.diff(kappa) / kappa[:-1]
    N = len(kappa) + 2

    # PARAMETER: alpha, S
    linregress = stats.linregress(R[:-1], -np.log(np.abs(dR/h)))
    alpha = linregress.slope
    S = np.exp(-linregress.intercept / (alpha - 1))

    if set_alpha is not None:
        print(f'Alg:\talpha = {alpha},\nUsr:\talpha = {set_alpha}')
        alpha = set_alpha
        intercept = -np.average(np.log(dR/h) + alpha * R[:-1])
        S = np.exp(-intercept / (alpha - 1))

    if filename is not None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(R[1:], -np.log(np.abs(dR/h)),
                c='black', lw=0.1, marker='o', markersize=1)
        X = np.linspace(np.min(R), np.max(R), len(R)*10)
        ax.plot(X, linregress.slope * X + linregress.intercept,
                c='red', lw=1)
        if set_alpha is not None:
            ax.plot(X, alpha * X + intercept,
                    c='orange', lw=1)
        fig.savefig(f'./output/debug/{filename}_internal_alpha_linreg.pdf')
        plt.close(fig)

    # PARAMETER: s0, z
    s0 = (np.average(1 / kappa**alpha) / alpha / S**alpha
          - 1 / alpha
          - h * (N - 1) / 2 / S)
    z = h / S

    return [alpha, S, s0, z]

def param_external(x, y, x_lac, y_lac, filename=None):
    # PARAMETER: phi (= average(theta - theta_lac))
    phi = np.angle((np.diff(x) + np.diff(y) * 1j)
                   / (np.diff(x_lac) + np.diff(y_lac) * 1j))
    for i in range(1, len(phi)):
        if phi[i] - phi[i-1] > np.pi:
            phi[i] -= 2*np.pi
            continue
        if phi[i-1] - phi[i] > np.pi:
            phi[i] += 2*np.pi
            continue
    # # ******** BEGIN plot rotation angle ********
    if filename is not None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(len(x)-1), phi)
        fig.savefig(f'./output/debug/{filename}_rotation.pdf')
        plt.close(fig)
    # # ******** END plot ********
    phi = np.average(phi)

    # PARAMETER: x0, y0
    xaux = np.array(x_lac)
    yaux = np.array(y_lac)
    curve.simtransf(xaux, yaux, 1, phi, 0, 0)

    x0 = np.average(x - xaux)
    y0 = np.average(y - yaux)

    return [phi, x0, y0]

def param_guess(gammax, gammay, param,
                err=1e-6, debug=None, set_alpha=None, filename=None):
    """First guess of LAC parameters.

    Arguments
    ---------
    ...
    filename : str, optional
        Filename used to save the debug figures to the file path
        (``./output/debug/{filename}...``). If ``filename`` is not ``None``
        the figures are save independently of argument ``debug`` being ``None``
        or not.

    debug : matplotlib.pyplot.Figure, optional
        If is not ``None`` plot the guessed LAC.
    """

    x = np.array(gammax)
    y = np.array(gammay)

    N = len(x)
    param.N = N

    [kappa, h] = curve.curvature(x, y, err=err)
    param.h = h
    L = h * (N - 1)
    param.L = L

    # # ******** BEGIN plot curvature ********
    if filename is not None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(N-2), kappa,
                c='blue', lw=0.5, marker='x', markersize=0.7)
        fig.savefig(f'./output/debug/{filename}_curvature.pdf')
        plt.close(fig)
    # # ******** END plot ********

    # NOTE: Guess case number base on the average curvature and its derivative
    val = np.sign(np.average(kappa))
    der = np.sign(np.average(np.diff(kappa)))
    case = 0
    if val < 0:
        if der < 0:
            case = 1
            kappa = -kappa[::-1]
        elif der > 0:
            case = 2
            kappa = -kappa
    elif val > 0:
        if der > 0:
            case = 3
            kappa = kappa[::-1]
    param.case = case
    curve.casetransf(x, y, case)

    # PARAMETERS: alpha, S, s0, ds (= z)
    [alpha, S, s0, ds] = param_internal(kappa, h, L,
                                        set_alpha=set_alpha,
                                        filename=filename)
    param.alpha = alpha
    param.S = S
    param.s0 = s0
    param.ds = ds
    param.l = ds * (N - 1)

    # PARAMETERS: phi, x0, y0
    param.phi = 0
    param.x0 = 0
    param.y0 = 0

    [x_lac, y_lac] = param.get_curve(case0=True)

    [phi, x0, y0] = param_external(x, y,
                                   x_lac, y_lac,
                                   filename=filename)
    param.phi = phi
    param.x0 = x0
    param.y0 = y0

    if debug is not None:
        x_debug = np.array(x_lac)
        y_debug = np.array(y_lac)
        curve.simtransf(x_debug, y_debug, 1, phi, x0, y0)
        debug.plot(x_debug, y_debug, c='pink', lw=2)
        debug.plot(x0, y0, 'x', c='orange', lw=2)
        param.print()
        print('***************************')

    return param.get_curve()
