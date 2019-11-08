import numpy as np
import pylab as pl
import os

from gnl_tensor import *
from viz import *
from io_waveform import *
from btensor import *


# setup
dev_x_path = os.path.join(os.path.dirname(__file__), 'gnlt/dev_x_connectom.nii.gz')
dev_y_path = os.path.join(os.path.dirname(__file__), 'gnlt/dev_y_connectom.nii.gz')
dev_z_path = os.path.join(os.path.dirname(__file__), 'gnlt/dev_z_connectom.nii.gz')

# compute voxelwise gradient non-linearity tensor
tensors = fsl_2_tensor(dev_x_path, dev_y_path, dev_z_path)

# load a brain mask
mask_img = nib.load(os.path.join(os.path.dirname(__file__), 'gnlt/mask.nii.gz'))
mask = mask_img.get_data()

# compute an index on the "amount" of distorsion for each gnl tensor
gnl_score = compute_gnl_score(tensors)

# Let's apply the distorsion on a Topgaard Spherical encoding
filename = os.path.join(os.path.dirname(__file__), 'waveform/sphere_GMAX_300.gp')
GMAX = 300e-3 # T/m

# read waveform file
gradient, t, dt = read_topgaard(filename, GMAX)

# Downsample for speed (takes ~10min otherwise)
ff = 9
gradient = gradient[::ff]
t = t[::ff]
dt = dt*ff

gradient_norm = np.linalg.norm(gradient, axis=1)

# plot gradient per axis and norm
pl.figure()
pl.subplot(2,1,1)
peraxisplot(t, gradient, title='Gradient', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.subplot(2,1,2)
plot(t, gradient_norm, label='', title='Gradient Norm', xlabel='time (s)', ylabel='gradient (T/m)', axhline=[0], axvline=[])
pl.show()

# sanity check on the undistorted
qt = compute_q_from_G(gradient, dt)
btensor = compute_B_from_q(qt, dt)
eigval, eigvec = get_btensor_eigen(btensor)
bval, bs, bp, bl = get_btensor_shape_topgaard(eigval)
print('Undistorted Values')
print('b = {:.2e} s/m^2'.format(bval))
print('b = {:.0f} s/mm^2'.format(1e-6*bval))
print('b = {:.3f} ms/um^2'.format(1e-9*bval))
print('sphere = {:.2f}'.format(bs))
print('planar = {:.2f}'.format(bp))
print('linear = {:.2f}'.format(bl))

# compute the voxelwise distorted
nbVox = np.sum(mask)
loopIt = 0
print('Distorting waveform')
dist_tensors = np.zeros_like(tensors)
for idx in np.ndindex(tensors.shape[:3]):
	if mask[idx]:
		if not (loopIt%5000):
			print('{} / {}'.format(loopIt, nbVox))
		loopIt += 1
		# grad the GNL tensor
		L = tensors[idx]
		# distort G
		dist_gradient = distort_G(gradient, L)
		# compute B-tensor
		dist_qt = compute_q_from_G(dist_gradient, dt)
		dist_tensors[idx] = compute_B_from_q(dist_qt, dt)

# compute metrics
loopIt = 0
print('Computing metrics')
dist_bval = np.zeros(tensors.shape[:3])
dist_bs = np.zeros(tensors.shape[:3])
dist_bp = np.zeros(tensors.shape[:3])
dist_bl = np.zeros(tensors.shape[:3])
for idx in np.ndindex(tensors.shape[:3]):
	if mask[idx]:
		if not (loopIt%5000):
			print('{} / {}'.format(loopIt, nbVox))
		loopIt += 1
		dist_btensor = dist_tensors[idx]
		dist_eigval, dist_eigvec = get_btensor_eigen(dist_btensor)
		tmp_bval, tmp_bs, tmp_bp, tmp_bl = get_btensor_shape_topgaard(dist_eigval)
		dist_bval[idx] = tmp_bval
		dist_bs[idx] = tmp_bs
		dist_bp[idx] = tmp_bp
		dist_bl[idx] = tmp_bl


# plot the scores
xyzplot(gnl_score, mask=mask, title='GNL scores', distval=0)
# pl.show()

# plot the metrics
xyzplot(dist_bval, mask=mask, title='bvalue', distval=bval)
# pl.show()
xyzplot(dist_bs, mask=mask, title='normalized b spherical', distval=bs)
# pl.show()
xyzplot(dist_bp, mask=mask, title='normalized b planar', distval=bp)
# pl.show()
xyzplot(dist_bl, mask=mask, title='normalized b linear', distval=bl)
pl.show()

