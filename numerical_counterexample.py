import numpy as np
import pylab as pl
import os

from io_waveform import *
from btensor import *
from viz import *




def block_gradient_1d(g, dt, t_a, delta, Delta, T):
	N = int(np.ceil((T / float(dt)))) + 1
	gradient = np.zeros(N)
	pos_t_a = int(np.ceil((t_a / float(dt))))
	len_delta = int(np.ceil((delta / float(dt))))
	len_Delta = int(np.ceil((Delta / float(dt))))
	gradient[pos_t_a:pos_t_a+len_delta+1] = g
	gradient[pos_t_a+len_Delta:pos_t_a+len_delta+len_Delta+1] = -g
	return gradient, np.linspace(0, T, N), dt





## GRADIENT 1
# block design for X partially overlapping in time with same block design for Y
# setup
GMAX = 100e-3 # T/m
T = 35e-3
delta = 5e-3
Delta = 20e-3
t_a = 5e-3
offset = 4e-3
dt = 0.01e-3


# read waveform file
gradient_x, t, dt = block_gradient_1d(GMAX, dt, t_a, delta, Delta, T)
gradient_y, t, dt = block_gradient_1d(GMAX, dt, t_a+offset, delta, Delta, T)
gradient = np.zeros((len(gradient_x), 3))
gradient[:,0] = gradient_x
gradient[:,1] = gradient_y
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







## GRADIENT 2
# same Delta, smaller delta, bigger GMAX
# block design for X not overlapping in time with same block design for Y
# setup
T2 = 35e-3
delta2 = 3e-3
Delta2 = 20e-3
t_a2 = 5e-3
offset = 4e-3
dt = 0.01e-3
# find GMAX to get the same b-value from: b = (GYRO*delta*G)**2 (Delta - delta/3)
GMAX2 = ((delta**2 * GMAX**2 * (Delta - delta/3.)) / (delta2**2 * (Delta2 - delta2/3.)))**0.5

# read waveform file
gradient_x2, t2, dt = block_gradient_1d(GMAX2, dt, t_a2, delta2, Delta2, T2)
gradient_y2, t2, dt = block_gradient_1d(GMAX2, dt, t_a2+offset, delta2, Delta2, T2)
gradient2 = np.zeros((len(gradient_x2), 3))
gradient2[:,0] = gradient_x2
gradient2[:,1] = gradient_y2
gradient_norm2 = np.linalg.norm(gradient2, axis=1)

# plot gradient per axis and norm
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t2, gradient2, title='Gradient', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.subplot(2,1,2)
plot(t2, gradient_norm2, label='', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.show()

# compute and plot q
qt2 = compute_q_from_G(gradient2, dt)
qt_norm2 = np.linalg.norm(qt2, axis=1)
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t2, qt2, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
pl.subplot(2,1,2)
plot(t2, qt_norm2, label='', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
pl.show()

# compute B-tensor and shape
btensor2 = compute_B_from_q(qt2, dt)
eigval2, eigvec2 = get_btensor_eigen(btensor2)
bval2, bs2, bp2, bl2 = get_btensor_shape_topgaard(eigval2)
print('B-tensor and shape:')
print(btensor2)
print('b = {:.2e} s/m^2'.format(bval2))
print('b = {:.0f} s/mm^2'.format(1e-6*bval2))
print('b = {:.3f} ms/um^2'.format(1e-9*bval2))
print('sphere = {:.2f}'.format(bs2))
print('planar = {:.2f}'.format(bp2))
print('linear = {:.2f}'.format(bl2))




# Introducing an example gradient linear tenser
print('Introducing distortion with tensor:')
L = np.array([[ 1. , -0.05, 0],
       		  [ -0.05,  1, 0],
       		  [0, 0, 1]])
print(L)
dist_gradient = distort_G(gradient, L)
dist_gradient_norm = np.linalg.norm(dist_gradient, axis=1)

dist_gradient2 = distort_G(gradient2, L)
dist_gradient_norm2 = np.linalg.norm(dist_gradient2, axis=1)


# quick visual comparison
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, gradient, title='Gradient vs distorted', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[], axvline=[])
peraxisplot(t, dist_gradient, title='Gradient vs distorted', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[], extra_label='dist ')
pl.subplot(2,1,2)
plot(t, gradient_norm, label='desired', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
plot(t, dist_gradient_norm, label='actual', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.show()


# quick visual comparison
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t2, gradient2, title='Gradient vs distorted', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[], axvline=[])
peraxisplot(t2, dist_gradient2, title='Gradient vs distorted', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[], extra_label='dist ')
pl.subplot(2,1,2)
plot(t2, gradient_norm2, label='desired', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
plot(t2, dist_gradient_norm2, label='actual', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.show()


# compute and plot q
dist_qt = compute_q_from_G(dist_gradient, dt)
dist_qt_norm = np.linalg.norm(dist_qt, axis=1)

dist_qt2 = compute_q_from_G(dist_gradient2, dt)
dist_qt_norm2 = np.linalg.norm(dist_qt2, axis=1)

pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, qt, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
peraxisplot(t, dist_qt, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[], extra_label='dist ')
pl.subplot(2,1,2)
plot(t, qt_norm, label='desired', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
plot(t, dist_qt_norm, label='actual', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
pl.show()


pl.figure()
pl.subplot(2,1,1)
peraxisplot(t2, qt2, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
peraxisplot(t2, dist_qt2, title='q-vector', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[], extra_label='dist ')
pl.subplot(2,1,2)
plot(t2, qt_norm2, label='desired', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
plot(t2, dist_qt_norm2, label='actual', title='q-vector Norm', xlabel='time (s)', ylabel='q (1/m)', axhline=[0], axvline=[])
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

# compute B-tensor and shape
dist_btensor2 = compute_B_from_q(dist_qt2, dt)
dist_eigval2, dist_eigvec2 = get_btensor_eigen(dist_btensor2)
dist_bval2, dist_bs2, dist_bp2, dist_bl2 = get_btensor_shape_topgaard(dist_eigval2)
print('B-tensor and shape:')
print(dist_btensor2)
print('b = {:.2e} s/m^2'.format(dist_bval2))
print('b = {:.0f} s/mm^2'.format(1e-6*dist_bval2))
print('b = {:.3f} ms/um^2'.format(1e-9*dist_bval2))
print('sphere = {:.2f}'.format(dist_bs2))
print('planar = {:.2f}'.format(dist_bp2))
print('linear = {:.2f}'.format(dist_bl2))










