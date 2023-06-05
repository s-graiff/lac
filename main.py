#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

# Description
# -----------
# Input curve is assumed to be a constant step size discretization of a
# continuous curve. The input curve is split into segments with monotonic radius
# of curvature, then each segment is approximated by an LAC independently from
# the others.
#
# Input
# -----
# ``./input/{FILENAME_EXT}`` : 2D curve with constant step size
#
# Output
# ------
# ``./output/{FILENAME}-curve-curvature-split.pdf``
# ``./output/{FILENAME}-curve-split-guess.pdf``
# ``./output/{FILENAME}-curve-ipopt-result-segment-{i}.pdf``
# ``./output/{FILENAME}-param-segment-{i}.txt``
# ``./output/{FILENAME}-segment-{i}_in.txt``
# ``./output/{FILENAME}-segment-{i}_guess.txt``
# ``./output/{FILENAME}-segment-{i}_out.txt``
# ``./output/{FILENAME}-ipopt-segment-{i}.txt``
# ``./output/{FILENAME}-curve-ipopt-result-all.pdf``
# and debug files as ``./output/debug/{FILENAME}-segment-{i}*``
# ------------------------------------------------------------------------------

import sys
import os
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import cyipopt as ipopt

# Local imports:
import lac
import optim_obj
import curve

# SECTION: Split input curve at the points where its curvature (kappa) or the
#          derivative of the curvature changes sign.
# ------------------------------------------------------------------------------
FILENAME_EXT = 'synthetic/syn0201.txt'
FILENAME = os.path.splitext(FILENAME_EXT)[0]

input_curve_txt = np.loadtxt(f'./input/{FILENAME_EXT}')

x = input_curve_txt[:, 0]
y = input_curve_txt[:, 1]

N = len(x)
[kappa, ds] = curve.curvature(x, y)
kappa_aux = interpolate.CubicSpline(range(N-2), kappa)
deriv_kappa_aux = kappa_aux.derivative()
# Alternatively,
# deriv_kappa_aux = interpolate.CubicSpline(range(N-3), np.diff(kappa))

segment_idx = np.sort(
    np.append(1 + kappa_aux.roots(extrapolate=False).astype(np.int16),
              1 + deriv_kappa_aux.roots(extrapolate=False).astype(np.int16)))
segment_count = len(segment_idx) + 1
x_split = np.split(x, segment_idx)
y_split = np.split(y, segment_idx)

# PLOT (Input curve, curvature and segment limits)
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Input curve')
ax.tick_params(left=False,
               bottom=False,
               labelleft=False,
               labelbottom=False)
ax.plot(x, y, 'black')
for i in range(segment_count):
    ax.plot(x_split[i][0], y_split[i][0], '|r')
    ax.plot(x_split[i][-1], y_split[i][-1], '.r')
ax = fig.add_subplot(2, 1, 2)
ax.set_ylabel('curvature')
ax.set_xlabel('arc length')
ax.plot(ds * range(N-2), kappa, 'black')
for i in range(len(segment_idx)):
    ax.plot(ds * segment_idx[i], kappa[segment_idx[i]], '. b')
fig.savefig(f'./output/{FILENAME}-curve-curvature-split.pdf')

# PLOT (Create and plot initial guess for all segments)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Input curve and initial guess')
ax.set_aspect('equal')
ax.plot(x, y, 'black')

for i in range(len(segment_idx)):
    ax.plot(x[segment_idx[i]], y[segment_idx[i]], '.b')

# Initialize Parameter object for each segment, then compute initial guess.
parameters = [lac.Parameter() for _ in range(segment_count)]
for i in range(segment_count):
    [xlac, ylac] = lac.param_guess(x_split[i], y_split[i],
                                   parameters[i],
                                   debug=None,
                                   filename=f'{FILENAME}-segment-{i}')
    ax.plot(xlac, ylac, 'red')

custom_lines = [Line2D([0], [0], color='black', lw=1, label='Input'),
                Line2D([0], [0], color='red', lw=1, label='Guess')]
ax.legend(handles=custom_lines, loc='upper left')
fig.savefig(f'./output/{FILENAME}-curve-split-guess.pdf')

# SECTION: IPOPT module
#          p = (alpha, s0,  l, S, phi, x0, y0) : continuous
#          p = (alpha, s0, ds, h, phi, x0, y0) : discretization
# ------------------------------------------------------------------------------
fig_all, ax_all = plt.subplots()
ax_all.set_aspect('equal')
ax_all.axis('off')
ax_all.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
for i in range(segment_count):
    x_input = np.array(x_split[i])
    y_input = np.array(y_split[i])

    # Input curve is transformed to case 0, to be used by the IPOPT module.
    curve.casetransf(x_input, y_input, parameters[i].case)

    p_ini = parameters[i].get_ipopt_style()

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
    nlp.add_option('max_iter', 1000)
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('mu_strategy', 'monotone')
    nlp.add_option('tol', 1e-3)
    nlp.add_option('acceptable_tol', 0.1)  # default = 1e-6
    nlp.add_option('nlp_scaling_method', 'none')
    nlp.add_option('output_file', f'./output/{FILENAME}-ipopt-segment-{i}.txt')

    # Run IPOPT:
    p_fin, info = nlp.solve(x=p_ini)
    print('(IPOPT computation ended)')

    # PLOT (input, guess, result)
    print('Guess Parameters:')
    parameters[i].printall()
    print('\nOutput Parameters:')
    ipopt_output = lac.Parameter()
    ipopt_output.set_ipopt_style(p_fin, parameters[i].N, parameters[i].case)
    ipopt_output.printall()

    [x_guess, y_guess] = parameters[i].get_curve()
    [x_output, y_output] = ipopt_output.get_curve()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.tick_params(left=False,
                   bottom=False,
                   labelleft=False,
                   labelbottom=False)
    curve.casetransf(x_input, y_input, parameters[i].case)
    ax.plot(x_input, y_input, c='black', lw=0.7,
            label='Input', marker='x', markersize=3)
    ax.plot(x_guess, y_guess, c='blue', lw=0.5,
            label='Guess', marker='o', markersize=2)
    ax.plot(x_output, y_output, c='red', lw=0.5,
            label='Output', marker='o', markersize=2)
    ax.legend()
    table_subplot = fig.add_subplot(1, 2, 2)
    table_subplot.axis('off')
    table_data = [['alpha', '%.3f' % parameters[i].alpha],
                  ['S', '%.3f' % parameters[i].S],
                  ['s0', '%.3f' % parameters[i].s0],
                  ['l', '%.3f' % parameters[i].l],
                  ['phi', '%.3f' % parameters[i].phi],
                  ['x0', '%.3f' % parameters[i].x0],
                  ['y0', '%.3f' % parameters[i].y0]]
    table = plt.table(cellText=table_data,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width((-1, 0, 1, 2, 3))
    fig.savefig(f'./output/{FILENAME}-curve-ipopt-result-segment-{i}.pdf')

    ax_all.plot(x_input, y_input, c='black', lw=0.7, marker='x', markersize=3)
    ax_all.plot(x_guess, y_guess, c='blue', lw=0.5, marker='o', markersize=2)
    ax_all.plot(x_output, y_output, c='red', lw=0.5, marker='o', markersize=2)

    with open(f'./output/{FILENAME}-param-segment-{i}.txt', 'w') as FILE:
        stdout = sys.stdout
        sys.stdout = FILE
        print('Guess Parameters:')
        parameters[i].printall()
        print('\nOutput Parameters:')
        ipopt_output.printall()
        sys.stdout = stdout

    with open(f'./output/{FILENAME}-segment-{i}_in.txt', 'w') as FILE:
        np.savetxt(FILE, np.column_stack((x_input, y_input)),
                   fmt='%.23f',
                   delimiter='\t')
    with open(f'./output/{FILENAME}-segment-{i}_guess.txt', 'w') as FILE:
        np.savetxt(FILE, np.column_stack((x_guess, y_guess)),
                   fmt='%.23f',
                   delimiter='\t')
    with open(f'./output/{FILENAME}-segment-{i}_out.txt', 'w') as FILE:
        np.savetxt(FILE, np.column_stack((x_output, y_output)),
                   fmt='%.23f',
                   delimiter='\t')

    if i + 1 == segment_count:
        print(f'Finished segment {i+1} out of {segment_count}.')
        continue

    prompt = input(f'Finished segment {i+1} out of {segment_count}. '
                   'Continue? (y/N) ')
    while prompt not in {'y', 'Y', 'n', 'N', ''}:
        prompt = input('(y/N) ')
    if prompt in {'y', 'Y'}:
        continue
    else:
        break

fig_all.savefig(f'./output/{FILENAME}-curve-ipopt-result-all.pdf')
# ------------------------------------------------------------------------------
