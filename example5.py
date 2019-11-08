import numpy as np
import pylab as pl
import os

from io_waveform import *
from btensor import *
from viz import *


# simple import and viz with gradient non-linearity distortion
# setup
filename = os.path.join(os.path.dirname(__file__), 'waveform/sphere_DUR_20_10_20_GMAX_300.txt')
GMAX = 300e-3 # T/m
DURATION_LEFT = 20e-3 # s
DURATION_PAUSE = 10e-3 # s
DURATION_RIGHT = 20e-3 # s

# read waveform file
gradient_low, t_low, dt_low = read_NOW(filename, DURATION_LEFT, DURATION_PAUSE, DURATION_RIGHT, GMAX)
gradient_low_norm = np.linalg.norm(gradient_low, axis=1)

# resampling low time-resolution gradient to >=1000 points for proper numerical sumation
gradient, t, dt = resample_waveform_equi(gradient_low, t_low, minN=1000)

# compute and plot q
qt = compute_q_from_G(gradient, dt)
qt_norm = np.linalg.norm(qt, axis=1)


# compute B-tensor and shape
btensor = compute_B_from_q(qt, dt)
eigval, eigvec = get_btensor_eigen(btensor)
bval, bs, bp, bl = get_btensor_shape_topgaard(eigval)
print('B-tensor and shape:')
print(btensor)
print('b = {:.2e} s/m^2'.format(bval))
print('b = {:.0f} s/mm^2'.format(1e-6*bval))
print('b = {:.3f} ms/um^2'.format(1e-9*bval))
print('sphere = {:.2f}'.format(bs))
print('planar = {:.2f}'.format(bp))
print('linear = {:.2f}'.format(bl))




# Introducing RANDOM gradient linear tensor
# with diagonal \in [0.9, 1.1] and off-diagonal \in [-0.05, 0.05]
print('Introducing random distortion with tensor:')

Lxx = 0.9 + 0.2*np.random.rand()
Lyy = 0.9 + 0.2*np.random.rand()
Lzz = 0.9 + 0.2*np.random.rand()

Lxy = -0.05 + 0.1*np.random.rand()
Lyx = -0.05 + 0.1*np.random.rand()
Lxz = -0.05 + 0.1*np.random.rand()
Lzx = -0.05 + 0.1*np.random.rand()
Lyz = -0.05 + 0.1*np.random.rand()
Lzy = -0.05 + 0.1*np.random.rand()

L = np.array([[Lxx, Lxy, Lxz],[Lyx, Lyy, Lyz],[Lzx, Lzy, Lzz]])
print(L)
dist_gradient = distort_G(gradient, L)
gradient_norm = np.linalg.norm(gradient, axis=1)
dist_gradient_norm = np.linalg.norm(dist_gradient, axis=1)


# compute and plot q
dist_qt = compute_q_from_G(dist_gradient, dt)
dist_qt_norm = np.linalg.norm(dist_qt, axis=1)

# compute B-tensor and shape
dist_btensor = compute_B_from_q(dist_qt, dt)
dist_eigval, dist_eigvec = get_btensor_eigen(dist_btensor)
dist_bval, dist_bs, dist_bp, dist_bl = get_btensor_shape_topgaard(dist_eigval)
print('\nCOMPUTED B-tensor and shape:')
print(dist_btensor)
print('b = {:.2e} s/m^2'.format(dist_bval))
print('b = {:.0f} s/mm^2'.format(1e-6*dist_bval))
print('b = {:.3f} ms/um^2'.format(1e-9*dist_bval))
print('sphere = {:.2f}'.format(dist_bs))
print('planar = {:.2f}'.format(dist_bp))
print('linear = {:.2f}'.format(dist_bl))


dist_btensor_p = predict_B_from_L(btensor, L)
dist_eigval_p, dist_eigvec_p = get_btensor_eigen(dist_btensor_p)
dist_bval_p, dist_bs_p, dist_bp_p, dist_bl_p = get_btensor_shape_topgaard(dist_eigval_p)
print('\nPREDICTED B-tensor and shape:')
print(dist_btensor_p)
print('b = {:.2e} s/m^2'.format(dist_bval_p))
print('b = {:.0f} s/mm^2'.format(1e-6*dist_bval_p))
print('b = {:.3f} ms/um^2'.format(1e-9*dist_bval_p))
print('sphere = {:.2f}'.format(dist_bs_p))
print('planar = {:.2f}'.format(dist_bp_p))
print('linear = {:.2f}'.format(dist_bl_p))


print('\nCOMPUTED minus PREDICTED')
print(dist_btensor - dist_btensor_p)
