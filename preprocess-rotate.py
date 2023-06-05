#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

# Description
# -----------
# Rotate raw input data into XY-plane.
# Plot raw input data, and rotated data.
# Plot curvature of raw input data.
# Save rotated curve as Numpy file.
#
# Input
# -----
# ``./input/prius/ps_{xx}.txt``
#
# Output
# ------
# ``./input/prius/preprocess/ps_{xx}-rotated.npy`` : Input of main_prius.py
# ``./input/prius/preprocess/ps_ALL-raw-rotated.pdf``
# ``./input/prius/preprocess/ps_ALL-raw-curvature.pdf``
# ------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt

# Local import:
import curve

# Render text in figures using LaTeX:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Serif"]
})

# Normal vector and its polar and azimut angle obtained from 3D software:
N_x = -0.99856279183126328558018
N_y = -0.05357678971846480814323
N_z = 0.00137053771081513875094
N = np.array([N_x, N_y, N_z])

N_xy = np.sqrt(N_x * N_x + N_y * N_y)
N_xyz = np.sqrt(N_xy * N_xy + N_z * N_z)

cos_polar = N_x / N_xy
sin_polar = N_y / N_xy

cos_azimu = N_z / N_xyz
sin_azimu = N_xy / N_xyz

# Rotation matrix: '-polar_angle' along Z axis
Rot_polar = np.array([[cos_polar, sin_polar, 0],
                      [-sin_polar, cos_polar, 0],
                      [0, 0, 1]])
# Rotation matrix: '-azimu_angle' along Y axis
Rot_azimu = np.array([[cos_azimu, 0, -sin_azimu],
                      [0, 1, 0],
                      [sin_azimu, 0, cos_azimu]])

assert np.allclose([0, 0, 1], Rot_azimu @ (Rot_polar @ N))

# SECTION: Rotate, plot and save input curves
# ------------------------------------------------------------------------------
fig = plt.figure()
fig.suptitle('All input curves, raw data')
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_aspect('auto')
ax3d.axis('off')
ax3d.set_title('Raw data')
ax3d.tick_params(left=False,
                 bottom=False,
                 labelleft=False,
                 labelbottom=False)
ax = fig.add_subplot(1, 2, 2)
ax.set_aspect('auto')
ax.axis('off')
ax.set_title('Projected curves (not to scale)')
ax.tick_params(left=False,
               bottom=False,
               labelleft=False,
               labelbottom=False)

fig2 = plt.figure()
fig2.subplots_adjust(left=0.1,
                     bottom=0.1,
                     right=0.9,
                     top=0.9,
                     wspace=0.1,
                     hspace=0.4)

for xx in range(5):
    print(f'\nAnalyzing file "ps_{xx+1}.txt".', end=' ')
    input_curve_txt = np.loadtxt(f'./input/prius/ps_{xx+1}.txt')
    x = input_curve_txt[:, 0]
    y = input_curve_txt[:, 1]
    z = input_curve_txt[:, 2]

    print('Rotated into:')
    [x_rot, y_rot, z_rot] = Rot_azimu @ (Rot_polar @ np.array([x, y, z]))
    [kappa, ds] = curve.curvature_3point(x_rot, y_rot)

    print('----------------------------------------')
    print(' axis  |      mean     | std. deviation')
    print('-------|---------------|----------------')
    print(f'  x    |{np.average(x_rot):12.5f}   |{np.std(x_rot):12.5f}')
    print(f'  y    |{np.average(y_rot):12.5f}   |{np.std(y_rot):12.5f}')
    print(f'  z    |{np.average(z_rot):12.5f}   |{np.std(z_rot):12.1e}')
    print('----------------------------------------')

    ax.plot(x_rot, y_rot, lw=0.1, marker='o', markersize=0.1)
    ax3d.plot(x, y, z, lw=0.1, marker='o', markersize=0.1)

    ax2 = fig2.add_subplot(5, 2, xx*2+1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'\\verb|ps_{xx+1}|')
    ax2.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    ax2.plot(y_rot, x_rot, c='black', lw=0.5)
    ax2 = fig2.add_subplot(5, 2, xx*2+2)
    ax2.set_ylim([-0.02, 0.02])
    ax2.tick_params(left=False,
                    right=True,
                    bottom=False,
                    labelleft=False,
                    labelright=True,
                    labelbottom=False)
    ax2.plot(np.cumsum(ds[:-1]), kappa, c='black', lw=0.5)

    with open(f'./input/prius/preprocess/ps_{xx+1}-rotated.npy', 'wb') as FILE:
        np.save(FILE, np.array([x_rot, y_rot, z_rot]))

fig.savefig('./input/prius/preprocess/ps_ALL-raw-rotated.pdf')

ax2.set_xlabel('arc length')
fig2.savefig('./input/prius/preprocess/ps_ALL-raw-curvature.pdf')
# ------------------------------------------------------------------------------
