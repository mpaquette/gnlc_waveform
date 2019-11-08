import numpy as np
import pylab as pl
import os

from io_waveform import *
from btensor import *
from viz import *


# simple import and viz with gradient non-linearity distortion
# setup
filename = os.path.join(os.path.dirname(__file__), 'waveform/sphere_GMAX_300.gp')
GMAX = 300e-3 # T/m

# read waveform file
gradient, t, dt = read_topgaard(filename, GMAX)
gradient_norm = np.linalg.norm(gradient, axis=1)

# plot gradient per axis and norm
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, gradient, title='Gradient', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.subplot(2,1,2)
plot(t, gradient_norm, label='', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.show()

# compute and plot q
qt = compute_q_from_G(gradient, dt)
qt_norm = np.linalg.norm(qt, axis=1)
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, qt, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
pl.subplot(2,1,2)
plot(t, qt_norm, label='', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
pl.show()

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

# Introducing an example gradient linear tenser
print('Introducing distortion with tensor:')
L = np.array([[ 1.0150782 ,  0.03378296,  0.01141613],
       		  [ 0.03415553,  1.0509096 ,  0.00681687],
       		  [-0.01310857, -0.01045736,  1.0371757 ]])
print(L)
dist_gradient = distort_G(gradient, L)
gradient_norm = np.linalg.norm(gradient, axis=1)
dist_gradient_norm = np.linalg.norm(dist_gradient, axis=1)

# quick visual comparison
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, gradient, title='Gradient vs distorted', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[], axvline=[])
peraxisplot(t, dist_gradient, title='Gradient vs distorted', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[], extra_label='dist ')
pl.subplot(2,1,2)
plot(t, gradient_norm, label='desired', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
plot(t, dist_gradient_norm, label='actual', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.show()

# compute and plot q
dist_qt = compute_q_from_G(dist_gradient, dt)
dist_qt_norm = np.linalg.norm(dist_qt, axis=1)
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, qt, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
peraxisplot(t, dist_qt, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[], extra_label='dist ')
pl.subplot(2,1,2)
plot(t, qt_norm, label='desired', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
plot(t, dist_qt_norm, label='actual', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
pl.show()

# compute B-tensor and shape
dist_btensor = compute_B_from_q(dist_qt, dt)
dist_eigval, dist_eigvec = get_btensor_eigen(dist_btensor)
dist_bval, dist_bs, dist_bp, dist_bl = get_btensor_shape_topgaard(dist_eigval)
print('B-tensor and shape:')
print(dist_btensor)
print('b = {:.2e} s/m^2'.format(dist_bval))
print('b = {:.0f} s/mm^2'.format(1e-6*dist_bval))
print('b = {:.3f} ms/um^2'.format(1e-9*dist_bval))
print('sphere = {:.2f}'.format(dist_bs))
print('planar = {:.2f}'.format(dist_bp))
print('linear = {:.2f}'.format(dist_bl))



