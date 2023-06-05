#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

def arclen_curve(x, y, s=None, *, N=None, h=None, dtmax=None, order=3):
    """Constant step size discretization of a discrete curve."""

    if N is None:
        N = len(x)

    if len(x) != len(y):
        raise ValueError(f'len(x) != len(y) ({len(x)} != {len(y)})')

    dx = np.diff(x)  # Note that x[i] = x[i - 1] + dx[i] (Size = N - 1)
    dy = np.diff(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    if s is None:
        s = np.append(0, np.cumsum(ds))  # Arch length parameter (Size = N)
    L = np.sum(ds)                       # Total length

    # Cubic Spline obtained from input:
    xsp = interpolate.UnivariateSpline(s, x, k=order)
    ysp = interpolate.UnivariateSpline(s, y, k=order)

    if h is None:
        h = L / N                        # Step length
    if dtmax is None:
        dtmax = 5*h

    def fn(t0, dt):
        return (xsp(t0 + dt) - xsp(t0))**2 + (ysp(t0 + dt) - ysp(t0))**2 - h**2

    t = []
    t0 = 0
    while t0 < L:
        # Find the root of fn() in the interval [t0, t1]
        t1 = h * 0.95
        while fn(t0, t1) < 0:
            t1 += h/5
            if t1 > dtmax:
                for i in range(len(s)):
                    if s[i] > t0:
                        break
                while fn(t0, t1) < 0 and t0+t1 < L+dtmax:
                    t1 += h/10
                # # DEBUG:
                # print(f'h = {h}, t1/h = {t1/h}')
                # print(f'len(t) = {len(t)}, N = {N}')
                # fig, ax = plt.subplots()
                # sa = np.linspace(t0, t0+t1+h/10, 30)
                # sa0 = np.linspace(t0, t0+h, 30)
                # ax.plot(xsp(sa), ysp(sa), color='black', lw=1)
                # ax.plot(xsp(sa0), ysp(sa0), color='red', lw=0.6)
                # circle = plt.Circle((xsp(t0), ysp(t0)),
                #                     h, color='b', fill=False)
                # ax.add_patch(circle)
                # ax.plot(xsp(t0), ysp(t0), 'xb')
                # ax.plot(xsp(t0 + t1), ysp(t0 + t1), 'ob')
                # ax.plot(xsp(t0 + h), ysp(t0 + h), 'or')
                # ax.set_aspect('equal')
                # plt.show()
                raise Warning('(t1 > dtmax) Problem with circle/spline intersect. '
                              f'at index {i-1} out of {N}. Possible solution '
                              f'found with dtmax/h > {t1/h:.3f}. Currently: '
                              f'h = {h:.3f}, dtmax/h = {dtmax/h:.3f}.')
        dt = optimize.brentq(
            lambda t: fn(t0, t),
            a=0,
            b=t1,
            xtol=1e-6,
            rtol=1e-8)
        t = np.append(t, dt)
        t0 += dt

    return (xsp(np.append([0], np.cumsum(t))),
            ysp(np.append([0], np.cumsum(t))),
            h)


def curvature(x_input, y_input, *, err=1e-6):
    """Curvature of a constant step size curve.

    Compute the curvature from the circle tangent to middle point of 2
    adjacent segments.
    """

    dx = np.diff(x_input)
    dy = np.diff(y_input)
    h = np.sqrt(dx**2 + dy**2)

    if np.std(h) < err:
        h = np.average(h)
    else:
        raise Exception('Non constant step size, '
                        f'h = {np.average(h):.3f} ({np.std(h):.2e})')

    sinp = (dx[:-1] * dy[1:] - dx[1:] * dy[:-1])  # = det(Tn-1, Tn) * h**2
    cosp = (dx[1:] * dx[:-1] + dy[:-1] * dy[1:])  # = <Tn-1, Tn> * h**2
    kappa = 2 / h * sinp / (h**2 + cosp)

    return (kappa, h)

def curvature_3point(x_input, y_input):
    """3-point curvature 'kappa'."""

    dx = np.diff(x_input)
    dy = np.diff(y_input)
    ds = np.sqrt(dx**2 + dy**2)
    sinp = (dx[:-1] * dy[1:] - dx[1:] * dy[:-1])  # / (ds[1:] * ds[:-1])
    cosp = (dx[1:] * dx[:-1] + dy[:-1] * dy[1:])  # / (ds[1:] * ds[:-1])
    phi = np.arctan2(sinp, cosp)                  # deflection angle
    kappa = (
        2 * np.sin(phi)
        / np.sqrt((x_input[2:] - x_input[:-2])**2
                  + (y_input[2:] - y_input[:-2])**2))
    return (kappa, ds)

def casetransf(x_input, y_input, case):
    x = np.array(x_input)
    y = np.array(y_input)

    if case == 1:
        x = x[::-1]
        y = y[::-1]
    elif case == 2:
        [x, y] = [y, x]
    elif case == 3:
        x = x[::-1]
        y = y[::-1]
        [x, y] = [y, x]

    for i in range(len(x)):
        x_input[i] = x[i]
        y_input[i] = y[i]


def simtransf(x_input, y_input, S=10, phi=1.570796, x0=0, y0=0):
    """Apply similarity transformation to mutable input curve."""

    x = np.array(x_input)
    y = np.array(y_input)
    cp = np.cos(phi)
    sp = np.sin(phi)

    [x, y] = [x0 + S * (x * cp - y * sp),
              y0 + S * (x * sp + y * cp)]

    for i in range(len(x)):
        x_input[i] = x[i]
        y_input[i] = y[i]


def plotcurv(x, y):
    """Return matplotlib figure of the curvature function of a constant
    step size curve.
    """

    N = len(x)
    assert N == len(y)

    dx = np.diff(x)
    dy = np.diff(y)
    h = np.sqrt(dx**2 + dy**2)

    L = np.sum(h)
    h = np.average(h)

    sinp = (dx[:-1] * dy[1:] - dx[1:] * dy[:-1])  # / h**2
    cosp = (dx[1:] * dx[:-1] + dy[:-1] * dy[1:])  # / h**2
    kappa = 2 / h * sinp / (h**2 + cosp)
    # Alternatively,
    # kappa = (dx[:-1] * np.diff(y, 2) - dy[:-1] * np.diff(x, 2)) / h**3

    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(4, 1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    ax.plot(x, y, 'black')

    ax = fig.add_subplot(4, 1, 2)
    ax.tick_params(bottom=False,
                   labelbottom=False)
    ax.plot(np.linspace(0, L, N-2), kappa)

    ax = fig.add_subplot(4, 1, 3)
    ax.tick_params(bottom=False,
                   labelbottom=False)
    ax.plot(np.linspace(0, L, N-2), -kappa)

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(np.linspace(0, L, N-2), -kappa[::-1])
    return fig
