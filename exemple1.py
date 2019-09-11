import numpy as np
import pylab as pl
import os

from io_waveform import *
from btensor import *
from viz import *


# simple import and viz
# setup
filename = os.path.join(os.path.dirname(__file__), 'waveform/planar_GMAX_300.gp')
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
print(btensor)
print('b = {:.2e} s/m^2'.format(bval))
print('b = {:.0f} s/mm^2'.format(1e-6*bval))
print('b = {:.3f} ms/um^2'.format(1e-9*bval))
print('sphere = {:.2f}'.format(bs))
print('planar = {:.2f}'.format(bp))
print('linear = {:.2f}'.format(bl))

