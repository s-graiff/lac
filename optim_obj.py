#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License

# Description
# -----------
# Definition of the ipopt problem objective.
#
# Reference
# ---------
# .. [1] Documentation of `cyipopt <https://pypi.org/project/cyipopt/>`_ :
#        https://cyipopt.readthedocs.io/en/stable/reference.html#cyipopt.problem
# ------------------------------------------------------------------------------

import numpy as np

class L2Distance:
    """ IPOPT ``problem_obj`` object.

    Parameter order
    ---------------
    Cont curve : p = (alpha, s0, l, S, phi, x0, y0)
    Disc curve : p = (alpha, s0, ds, h, phi, x0, y0)
    """

    def __init__(self, x_input, y_input, p_initial):
        if len(x_input) != len(y_input):
            raise ValueError('ObjectiveData: different input length')
        self.x = x_input
        self.y = y_input
        self.N = len(x_input)

    def objective(self, p):
        alpha = p[0]
        s0 = p[1]
        ds = p[2]
        h = p[3]
        phi = p[4]
        x0 = p[5]
        y0 = p[6]

        lac_x = x0
        lac_y = y0

        f_x = x0 - self.x[0]
        f_y = y0 - self.y[0]

        F = (f_x * f_x + f_y * f_y) * 0.5

        s = s0
        # turning angle: Added abs() but we should check it is always positive:
        theta = (np.power(np.abs(1 + alpha * s), 1 - 1/alpha) - 1) / (alpha - 1)
        #                 ^^^^^^
        T_x = np.cos(theta + phi)
        T_y = np.sin(theta + phi)

        for n in np.arange(1, self.N):
            lac_x += h * T_x
            lac_y += h * T_y

            f_x = lac_x - self.x[n]
            f_y = lac_y - self.y[n]

            # Objective function:
            F += (f_x * f_x + f_y * f_y) * 0.5

            # UPDATE: theta, T_x, T_y for next loop
            s += ds
            # Added abs() but we should check it is always positive
            theta = (np.power(np.abs(1 + alpha*s), 1 - 1/alpha) - 1) / (alpha-1)
            #                 ^^^^^^
            T_x = np.cos(theta + phi)
            T_y = np.sin(theta + phi)

        return F

    def gradient(self, p):
        alpha = p[0]
        s0 = p[1]
        ds = p[2]
        h = p[3]
        phi = p[4]
        x0 = p[5]
        y0 = p[6]

        lac_x = x0
        lac_y = y0

        f_x = x0 - self.x[0]
        f_y = y0 - self.y[0]

        F = (f_x * f_x + f_y * f_y) * 0.5

        s = s0
        # absolute value is added by must be controlled by the domain of 's'
        sap = np.abs(1 + s * alpha)
        #     ^^^^^^
        am = alpha - 1
        am2 = am * am
        ainv = 1/alpha

        kappa = np.power(sap, -ainv)                # LAC curvature
        theta = (np.power(sap, 1 - ainv) - 1) / am  # LAC turning angle

        # Derivative of the turning angle w.r.t. alpha:
        alpha_deriv_theta = 1 / am2 + (
            kappa
            * (am * sap * np.log(sap) - alpha * (sap + am * (s + 1)))
            / (alpha * alpha * am2))

        # Tangent vectors:
        T_x = np.cos(theta + phi)
        T_y = np.sin(theta + phi)

        # Partial derivative of the curve 'x' and 'y' components  w.r.t. to the
        # parameters p. Note that:
        # pderiv_lac_x[i] = 0 for all i != 5 (p != x0),
        # pderiv_lac_y[i] = 0 for all i != 6 (p != y0).
        pderiv_lac_x = np.zeros(7)
        pderiv_lac_y = np.zeros(7)
        pderiv_lac_x[5] = 1
        pderiv_lac_y[6] = 1

        # Gradient of the objective function:
        # NOTE: gradF_aux[i] = 0 for all i != 5 and i!= 6.
        gradF = np.zeros(7)
        gradF[5] = f_x
        gradF[6] = f_y

        for n in np.arange(1, self.N):
            lac_x += h * T_x
            lac_y += h * T_y

            f_x = lac_x - self.x[n]
            f_y = lac_y - self.y[n]

            F += (f_x * f_x + f_y * f_y) * 0.5

            # NOTE: pderiv_lac_x[i] and pderiv_lac_y[i] don't change for
            # i = 5 (x0) and i = 6 (y0)
            pderiv_lac_x[0] -= h * T_y * alpha_deriv_theta
            pderiv_lac_y[0] += h * T_x * alpha_deriv_theta

            pderiv_lac_x[1] -= h * T_y * kappa
            pderiv_lac_y[1] += h * T_x * kappa

            pderiv_lac_x[2] -= h * T_y * kappa * n
            pderiv_lac_y[2] += h * T_x * kappa * n

            pderiv_lac_x[3] += T_x
            pderiv_lac_y[3] += T_y

            pderiv_lac_x[4] -= h * T_y
            pderiv_lac_y[4] += h * T_x

            # Gradient of the objective function:
            gradF += f_x * pderiv_lac_x + f_y * pderiv_lac_y

            # Update variables: kappa, theta, alpha_deriv_theta, T_x, T_y
            s += ds
            # Added abs() but we should check it is always positive
            sap = np.abs(1 + s * alpha)
            am = alpha - 1
            am2 = am * am
            ainv = 1/alpha

            kappa = np.power(sap, -ainv)
            theta = (np.power(sap, 1 - ainv) - 1) / am
            alpha_deriv_theta = 1 / am2 + (
                kappa
                * (am * sap * np.log(sap) - alpha * (sap + am * (s + 1)))
                / (alpha * alpha * am2))

            T_x = np.cos(theta + phi)
            T_y = np.sin(theta + phi)

        return np.array([
            gradF[0],           # pderiv_alpha
            gradF[1],           # pderiv_s0
            gradF[2],           # pderiv_ds
            gradF[3],           # pderiv_h
            gradF[4],           # pderiv_phi
            gradF[5],           # pderiv_x0
            gradF[6]            # pderiv_y0
        ])

    def constraints(self, p):
        """Constraint the domain of ``s``."""
        return np.array([1 + p[0] * p[1]])

    def jacobian(self, p):
        return np.array([[p[1], p[0]]])

    def jacobianstructure(self):
        return np.array([[0, 0], [0, 1]])
