#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

import numpy as np
import math

def pivot_dmax(x, y):
    """Return the index of the furthest point from the curve to the line segment
    connecting the initial and final points of the curve.
    """

    dmax2 = 0
    index = None
    for i in range(1, len(x) - 1):
        d2 = (x[i] * y[-1] - y[i] * x[-1]
              + x[0] * y[i] - y[0] * x[i]
              + x[-1] * y[0] - y[-1] * x[0]) ** 2
        d2 /= ((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

        if d2 > dmax2:
            dmax2 = d2
            index = i

    if index is None:
        raise ValueError('index is None')

    return (index, math.sqrt(dmax2))


def rdp_classic(x, y, epsilon):
    """Return curve obtained by applying the RDP algorithm.

    Reference
    ---------
    .. [1] `Ramer–Douglas–Peucker algorithm <https://w.wiki/3X7H>`_
    """

    if len(x) == 2:
        return np.array([[x[0], y[0]], [x[-1], y[-1]]])

    index, dist = pivot_dmax(x, y)

    if dist > epsilon:
        left = rdp_classic(x[0:index+1], y[0:index+1], epsilon)
        right = rdp_classic(x[index:], y[index:], epsilon)
        return np.append(left[0:-1], right, axis=0)
    else:
        return np.array([[x[0], y[0]], [x[-1], y[-1]]])


def rdp_pivots(x, y, epsilon, *, ini=0, fin=None):
    """Return only the curve indexes for the ``rdp_classic`` algorithm."""
    if fin is None:
        fin = len(x)
    if len(x) == 2:
        if ini == 0:
            return [0, fin-1]
        return [fin-1]

    pivot, dist = pivot_dmax(x[ini:fin], y[ini:fin])
    pivot += ini

    if dist > epsilon:
        left = rdp_pivots(x, y, epsilon,
                          ini=ini, fin=pivot+1)
        right = rdp_pivots(x, y, epsilon,
                           ini=pivot, fin=fin)
        return left + right
    else:
        if ini == 0:
            return [0, fin-1]
        return [fin-1]


def rdp_Npivots(x, y, N=5):
    """``rdp_pivots`` algorithm with fixed size for the output curve."""

    _, eps = pivot_dmax(x, y)
    pivots = rdp_pivots(x, y, eps)
    while len(pivots) < N:
        eps *= 0.9
        pivots = rdp_pivots(x, y, eps)
    return (pivots, eps)


def rdp(x, y, N=5):
    """``rdp_classic`` algorithm with fixed size for the output curve."""
    _, eps = pivot_dmax(x, y)
    res_aux = rdp_classic(x, y, eps)
    while len(res_aux) < N:
        eps *= 0.9
        res_aux = rdp_classic(x, y, eps)
    return (np.transpose(res_aux), eps)
