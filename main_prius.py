#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

# Description
# -----------
# Input curves are the raw-projected data (see NOTE below) of the Prius model.
# The input curve is firstly smoothen using pivots obtained from an RDP
# algorithm. Each smoothed curve is converted into a constant step size curve,
# then it is approximates by an LAC.
#
# [NOTE] Before running this code, run ``preprocess-rotate.py`` over the 3D data.
#
# Input
# -----
# ``./input/prius/preprocess/ps_{xx}-rotated.npy`` : 2D Numpy file
#
# Output
# ------
# ``./output/prius/ps_{xx}-h-kappa.pdf``
# ``./output/prius/ps_{xx}-param.txt``
# ``./output/prius/ps_{xx}-ipopt.txt``
# ``./output/prius/ps_ALL-h-initial-guess-alg.pdf``
# ``./output/prius/ps_ALL-ipopt-result.pdf``
# and debug files as ``./output/debug/prius/ps_{xx}*``
# ------------------------------------------------------------------------------

import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from operator import concat
import cyipopt as ipopt

# Local imports:
import lac
import optim_obj
import curve
import rdp

# Render text in figures using LaTeX:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"]
})

# IPOPT : bool\ True will run the optimization.
# TABLE : bool\ True will print the final output with the result plot.
# fig2 : figure\ Contains the input and first guess.
# fig_ipopt : figure\ Contains the input, first guess and ipopt result.
IPOPT = True
TABLE = True
fig2 = plt.figure()
fig2.subplots_adjust(left=0.1, wspace=0.1)
if IPOPT is True:
    fig_ipopt = plt.figure()
    fig_ipopt.subplots_adjust(left=0.1, wspace=0.1)

parameters_outlist = [lac.Parameter() for _ in range(5)]

for xx in range(5):
    address = f'prius/ps_{xx+1}'
    filename = f'ps_{xx+1}'
    print('----------------------------------------------------------------')
    print(f'Analyzing: {filename}')

    # Raw data (only projected in 2D):
    with open(f'./input/prius/preprocess/{filename}-rotated.npy', 'rb') as FILE:
        [x_raw, y_raw, _] = np.load(FILE)

    kappa_raw, ds_raw = curve.curvature_3point(x_raw, y_raw)

    # RDP interpolation:
    rdp_pivots, rdp_eps = rdp.rdp_Npivots(x_raw, y_raw, N=5)
    xrdp = x_raw[rdp_pivots]
    yrdp = y_raw[rdp_pivots]
    srdp = [sum(ds_raw[0:i]) for i in rdp_pivots]

    # Constant arc-length curve from RDP with size equal to raw data:
    [xh, yh, h] = curve.arclen_curve(xrdp, yrdp, srdp, N=len(x_raw))
    Nh = len(xh)
    kappa, h_aux = curve.curvature(xh, yh)
    assert np.isclose(h, h_aux)

    # PLOT (raw data, rdp pivots, smooth curve and curvature)
    fig = plt.figure()
    fig.suptitle(f'\\verb|{filename}-h|')
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.4)
    gs = fig.add_gridspec(1, 5)
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    ax.plot(x_raw, y_raw, c='black', lw=0, marker='o', markersize=1)
    ax.plot(xh, yh, c='cyan', lw=0.7)
    ax.plot(xh[0], yh[0], c='cyan', marker='o', markersize=5)
    ax.plot(xrdp, yrdp, c='red', lw=0, marker='x', markersize=5)
    ax = fig.add_subplot(gs[2:])
    ax.set_ylabel('curvature')
    ax.set_xlabel('arc length')
    ax.plot(h * range(Nh-2), kappa, c='black', lw=0.5, marker='o', markersize=0.5)
    fig.savefig(f'./output/{address}-h-kappa.pdf')
    plt.close(fig)

    # Initial Guess:
    parameters = lac.Parameter()
    [xlac, ylac] = lac.param_guess(xh, yh,
                                   parameters,
                                   debug=None,
                                   filename=address,
                                   set_alpha=None)
    parameters_outlist[xx] = parameters

    with open(f'./output/{address}-param.txt', 'w') as FILE:
        stdout = sys.stdout
        sys.stdout = FILE
        print('Guess Parameters:')
        parameters.printall()
        sys.stdout = stdout

    if IPOPT is False and TABLE is True:
        ax = plt.subplot2grid((1, 10), (0, xx), fig=fig2)
        ax.set_title(f'\\verb|{filename}|')
    else:
        ax = fig2.add_subplot(1, 5, xx+1)
        ax.set_title(f'\\verb|{filename}|\\\\$\\alpha ={parameters.alpha:.3f}$')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    ax.plot(x_raw, y_raw, c='black', lw=0, marker='o', markersize=1)
    ax.plot(xh, yh, c='cyan', lw=0.7)
    ax.plot(xlac, ylac, c='red', lw=0.5)

    if IPOPT is False:
        continue

    # SECTION: IPOPT module
    # --------------------------------------------------------------------------
    x_input = np.array(xh)
    y_input = np.array(yh)

    # Input curve is transformed to case 0, to be used by the IPOPT module.
    curve.casetransf(x_input, y_input, parameters.case)

    p_ini = parameters.get_ipopt_style()

    # Optimization variable bounds:
    eps = 0.1
    p_lower = np.array([-10e10, -10e10, 1e-10, 1e-10, -2*np.pi, -10e10, -10e10])
    p_upper = np.array([10e10, 10e10, 10e10, 10e10, 2*np.pi, 10e10, 10e10])
    for _ in range(len(p_ini)):
        p_lower[_] = np.maximum(p_ini[_] - abs(p_ini[_]) * eps, p_lower[_])
        p_upper[_] = np.minimum(p_ini[_] + abs(p_ini[_]) * eps, p_upper[_])

    # Optimization constraints bounds:
    C_lower = np.array([0])
    C_upper = np.array([10e10])

    # Create IPOPT problem:
    nlp = ipopt.Problem(
        n=len(p_ini),
        m=len(C_lower),
        problem_obj=optim_obj.L2Distance(x_input, y_input, p_ini),
        lb=p_lower,
        ub=p_upper,
        cl=C_lower,
        cu=C_upper
    )
    # nlp.add_option('derivative_test', 'first-order')
    # nlp.add_option('derivative_test_print_all', 'yes')
    nlp.add_option('max_iter', 300)  # TODO
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('mu_strategy', 'monotone')
    nlp.add_option('tol', 1e-3)
    nlp.add_option('acceptable_tol', 0.1)  # default = 1e-6
    nlp.add_option('output_file', f'./output/{address}-ipopt.txt')
    nlp.add_option('nlp_scaling_method', 'none')

    # Run IPOPT:
    p_fin, info = nlp.solve(x=p_ini)
    print('(IPOPT computation ended)')

    # PLOT (input, guess, result)
    print('Guess Parameters:')
    parameters.printall()
    print('\nOutput Parameters:')
    ipopt_output = lac.Parameter()
    ipopt_output.set_ipopt_style(p_fin, parameters.N, parameters.case)
    parameters_outlist[xx] = ipopt_output
    ipopt_output.printall()

    [x_guess, y_guess] = parameters.get_curve()
    [x_output, y_output] = ipopt_output.get_curve()
    curve.casetransf(x_input, y_input, parameters)

    ax_ipopt = plt.subplot2grid((1, 10), (0, xx), fig=fig_ipopt)
    ax_ipopt.set_title(f'\\verb|{filename}|')
    ax_ipopt.set_aspect('equal')
    ax_ipopt.axis('off')
    ax_ipopt.tick_params(left=False,
                         bottom=False,
                         labelleft=False,
                         labelbottom=False)
    ax_ipopt.plot(x_input, y_input, c='black', lw=0, marker='o', markersize=1)
    ax_ipopt.plot(x_guess, y_guess, c='blue', lw=0.7)
    ax_ipopt.plot(x_output, y_output, c='red', lw=0.5)

    with open(f'./output/{address}-param.txt', 'w') as FILE:
        stdout = sys.stdout
        sys.stdout = FILE
        print('Guess Parameters:')
        parameters.printall()
        print('\nOutput Parameters:')
        ipopt_output.printall()
        sys.stdout = stdout

custom_lines = [Line2D([0], [0], color='black', lw=0,
                       marker='o', markersize=1, label='Input'),
                Line2D([0], [0], color='red', lw=1, label='LAC-guess'),
                Line2D([0], [0], color='cyan', lw=1, label='RDP-h')]
ax.legend(handles=custom_lines, loc='best')

if TABLE is True:
    if IPOPT is True:
        table_subplot = plt.subplot2grid((1, 10), (0, 5), colspan=5, fig=fig_ipopt)
    else:
        table_subplot = plt.subplot2grid((1, 10), (0, 5), colspan=5, fig=fig2)

    table_subplot.axis('off')
    table_data = [[' ', 1, 2, 3, 4, 5],
                  concat(['alpha'],
                         ['%.2f' % parameters_outlist[i].alpha for i in range(5)]),
                  concat(['S'], ['%.2f' % parameters_outlist[i].S for i in range(5)]),
                  concat(['s0'], ['%.2f' % parameters_outlist[i].s0 for i in range(5)]),
                  concat(['l'], ['%.2f' % parameters_outlist[i].l for i in range(5)]),
                  concat(['phi'], ['%.2f' % parameters_outlist[i].phi for i in range(5)]),
                  concat(['x0'], ['%.2f' % parameters_outlist[i].x0 for i in range(5)]),
                  concat(['y0'], ['%.2f' % parameters_outlist[i].y0 for i in range(5)])]
    table = plt.table(cellText=table_data,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

fig2.suptitle('All INPUT, RDP-h and LAC-guess')
fig2.savefig('./output/prius/ps_ALL-h-initial-guess-alg.pdf')

if IPOPT is True:
    fig_ipopt.suptitle('All INPUT, GUESS and OUTPUT')
    fig_ipopt.savefig('./output/prius/ps_ALL-ipopt-result.pdf')
    plt.close(fig_ipopt)

plt.close(fig2)
# ------------------------------------------------------------------------------
