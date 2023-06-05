#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

# Description
# -----------
# Pre-processing of the curve (using RDP algorithm) to select structural nodes.
# Then this curve is used to generate a spline of order ``ORDER``.
#
# Input
# -----
# ``./input/prius/preprocess/ps_{xx}-rotated.npy``
#
# Output
# ------
# ``./input/prius/preprocess/ps_{xx}-rdp-h.pdf``
# ``./input/prius/preprocess/ps_ALL-raw-curvature-rdp-h.pdf``
# ------------------------------------------------------------------------------

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# Local imports:
import rdp
import curve

# Render text in figures using LaTeX:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"]
})

# NOTE:
#     fig2 : contains all curves and their curvatures
#     fig : contains a single curve and its curvature
fig2 = plt.figure()
fig2.subplots_adjust(left=0.1,
                     bottom=0.1,
                     right=0.9,
                     top=0.9,
                     wspace=0.1,
                     hspace=0.4)

for xx in range(5):
    FILENAME = f'ps_{xx+1}'
    print('----------------------------------------')
    print(f'Analyzing: {FILENAME}')

    with open(f'./input/prius/preprocess/{FILENAME}-rotated.npy', 'rb') as FILE:
        [x_raw, y_raw, _] = np.load(FILE)

    dx_raw = np.diff(x_raw)
    dy_raw = np.diff(y_raw)
    ds_raw = np.sqrt(dx_raw**2 + dy_raw**2)
    s_raw = np.append(0, np.cumsum(ds_raw))
    L_raw = np.sum(ds_raw)
    N_raw = len(x_raw)
    kappa_raw, _ = curve.curvature_3point(x_raw, y_raw)

    # RDP curve:
    N = 5                          # Length of the desired RDP curve
    rdp_pivots, rdp_eps = rdp.rdp_Npivots(x_raw, y_raw, N=N)
    xrdp = x_raw[rdp_pivots]
    yrdp = y_raw[rdp_pivots]
    srdp = s_raw[rdp_pivots]
    N = len(xrdp)                  # Update length

    # Spline from RDP curve:
    ORDER = 3                       # Spline order 'ORDER' <= 'N'
    xsp = interpolate.UnivariateSpline(srdp, xrdp, k=ORDER)
    ysp = interpolate.UnivariateSpline(srdp, yrdp, k=ORDER)

    # 'Discrete' spline from input arc-length:
    x = xsp(s_raw)                # Non-constant length
    y = ysp(s_raw)

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    L = np.sum(ds)
    s = np.append(0, np.cumsum(ds))
    kappa, _ = curve.curvature_3point(x, y)

    res = np.sum(
        np.sqrt((x[:-1] - x_raw[:-1]) ** 2 + (y[:-1] - y_raw[:-1]) ** 2)
        * np.sqrt(ds * ds_raw)
    )

    print(f'\t(RDP {N} points, spline order {ORDER})')
    print(f'Residual:\n\t{res / (L_raw ** 2) * 100:.3f}%')

    # Plot and save figure:
    fig = plt.figure()
    fig.suptitle(f'\\verb|{FILENAME}| (RDP {N} points, spline order {ORDER})')
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    ax.plot(x_raw, y_raw, c='black', lw=0, marker='o', markersize=1)
    ax.plot(x, y, c='green', lw=0.7)
    ax.plot(x[0], y[0], c='green', marker='o', markersize=5)
    ax.plot(xrdp, yrdp, c='red', lw=0, marker='x', markersize=5)

    ax = fig.add_subplot(2, 2, 2)
    ax.set_ylabel('curvature')
    ax.set_xlabel('arc length')
    ax.plot(s[1:-1], kappa, c='black', lw=0.5, marker='o', markersize=0.5)

    ax = fig.add_subplot(2, 2, 4)
    ax.axis('off')
    custom_lines = [Line2D([0], [0], color='black',
                           lw=0, marker='o', markersize=1, label='Input'),
                    Line2D([0], [0], color='red',
                           lw=0, marker='x', markersize=5, label='RDP'),
                    Line2D([0], [0], color='green',
                           lw=0.7, marker='o', markersize=5, label='Spline')]
    ax.legend(handles=custom_lines, loc='center')

    fig.savefig(f'./input/prius/preprocess/{FILENAME}-rdp-h.pdf')
    plt.close(fig)

    # -------------------------------------------------------------------------
    ax2 = fig2.add_subplot(5, 3, xx*3+1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'\\verb|ps_{xx+1}|')
    ax2.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    ax2.plot(y_raw, x_raw, c='black', lw=1)
    ax2.plot(y, x, c='red', lw=0.3)

    ax2 = fig2.add_subplot(5, 3, xx*3+2)
    ax2.set_ylim([-0.005, 0.005])
    ax2.tick_params(left=False,
                    right=False,
                    bottom=False,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False)
    if xx == 0:
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('(raw and smoothen data)')

    ax2.plot(np.cumsum(ds_raw[:-1]), kappa_raw, c='black', lw=0.3)
    ax2.axhline(y=0, c='blue', lw=0.7)
    ax2.plot(np.cumsum(ds[:-1]), kappa, c='red', lw=0.7)

    ax3 = fig2.add_subplot(5, 3, xx*3+3)
    yrange = np.max(kappa) - np.min(kappa)
    ax3.set_ylim([np.min(kappa) - 0.5 * yrange, np.max(kappa) + 0.5 * yrange])
    ax3.tick_params(left=False,
                    right=False,
                    bottom=False,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False)
    if xx == 0:
        ax3.xaxis.set_label_position('top')
        ax3.set_xlabel('(close-up: smoothen data)')

    ax3.plot(np.cumsum(ds_raw[:-1]), kappa_raw, c='grey', lw=0.1)
    ax3.plot(np.cumsum(ds[:-1]), kappa, c='red', lw=1)

ax2.set_xlabel('arc length')
ax3.set_xlabel('arc length')
fig2.savefig('./input/prius/preprocess/ps_ALL-raw-curvature-rdp-h.pdf')
# ------------------------------------------------------------------------------
