"""Finite AdamW transport at a fixed clipped gradient, in real arithmetic.

These functions evaluate the mathematical prediction in float64. They do not
implement training, differentiate a floating-point program, or certify its
roundoff. Native agreement is measured separately.
"""
import numpy as np


def fixed_gradient_step(theta, first, second, gradient, *, beta1, beta2,
                        counter, rate, epsilon, decay):
    if counter < 1 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError('Invalid moment clock')
    if epsilon <= 0 or np.any(np.asarray(second) < 0):
        raise ValueError('Invalid second moment or denominator offset')
    p, m, v, g = [np.asarray(x, dtype=np.float64)
                  for x in (theta, first, second, gradient)]
    mp = beta1*m + (1-beta1)*g
    vp = beta2*v + (1-beta2)*g*g
    pp = (1-rate*decay)*p - rate*mp/(1-beta1**counter)/(np.sqrt(vp/(1-beta2**counter))+epsilon)
    return pp, mp, vp


def pulse_parts(zero, plus, minus):
    return (np.asarray(plus)+np.asarray(minus))/2-np.asarray(zero), (np.asarray(plus)-np.asarray(minus))/2


def first_moment_displacement(first_delta, second_out, *, beta1, beta2,
                              counter, rate, epsilon):
    denominator = np.sqrt(np.asarray(second_out)/(1-beta2**counter))+epsilon
    return -rate*beta1*np.asarray(first_delta)/(1-beta1**counter)/denominator
