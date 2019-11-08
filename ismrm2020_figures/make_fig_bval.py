# Script to generate one of the ISMRM figure
import os

import numpy as np
import pylab as pl
import nibabel as nib

from time import time

import matplotlib as mpl
# mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

import string
import matplotlib.patches as mpatches

from gnlc_waveform.gnl_tensor import fsl_2_tensor
from gnlc_waveform.btensor import predict_B_list_from_L

from gnlc_waveform.btensor import get_btensor_eigen, get_btensor_shape_topgaard_list




# # file setup
datapath = '/data/pt_02015/190917_FWF_Meas_Connectom_GradientNonlinearities_Shift/'




# grab the voigt b-tensor
# [xx yy zz xy xz yz] * [1 1 1 sqrt(2) sqrt(2) sqrt(2)]
tmp = np.genfromtxt(os.path.join(os.path.dirname(__file__), '../waveform/bt384.txt'))
# swap b-tensor into [xx yy zz yz xz xy] ordering
btV = np.empty_like(tmp)
btV[:,:3] = tmp[:,:3]
btV[:,3] = tmp[:,5]
btV[:,4] = tmp[:,4]
btV[:,5] = tmp[:,3]
del tmp

# convert back to 2D from PROPER order
bt = np.empty((btV.shape[0],3,3))
bt[:,0,0] = btV[:,0]
bt[:,1,1] = btV[:,1]
bt[:,2,2] = btV[:,2]
bt[:,1,2] = btV[:,3] / np.sqrt(2)
bt[:,2,1] = btV[:,3] / np.sqrt(2)
bt[:,0,2] = btV[:,4] / np.sqrt(2)
bt[:,2,0] = btV[:,4] / np.sqrt(2)
bt[:,0,1] = btV[:,5] / np.sqrt(2)
bt[:,1,0] = btV[:,5] / np.sqrt(2)

# nicer units
bt *= 1e-9



from gnlc_waveform.gnl_tensor import fsl_2_tensor
from gnlc_waveform.btensor import predict_B_list_from_L


# setup GNL tensor
dev_x_path = os.path.join(os.path.dirname(__file__), '../gnlt/dev_x_connectom2.nii.gz')
dev_y_path = os.path.join(os.path.dirname(__file__), '../gnlt/dev_y_connectom2.nii.gz')
dev_z_path = os.path.join(os.path.dirname(__file__), '../gnlt/dev_z_connectom2.nii.gz')

# compute voxelwise gradient non-linearity tensor
tensors = fsl_2_tensor(dev_x_path, dev_y_path, dev_z_path)



# grab the mask
maskimg = nib.load(os.path.join(os.path.dirname(__file__), '../gnlt/mask2.nii.gz'))
mask = maskimg.get_data().astype(np.bool)

I3 = np.eye(3)
norms = np.zeros_like(mask, dtype=np.float)
for x in range(mask.shape[0]):
	for y in range(mask.shape[1]):
		for z in range(mask.shape[2]):
			if mask[x,y,z]:
				L = tensors[x,y,z]
				norms[x,y,z] = np.linalg.norm(L-I3)


# pl.figure()
# pl.hist(norms[mask], bins=100)
# pl.show()

# quick correction of something very wrong with the mask
mask2 = np.logical_and(mask, norms <= 0.2)

print(mask.sum())
print(mask2.sum())





bt_true = np.zeros((mask2.shape[0], mask2.shape[1], mask2.shape[2], bt.shape[0], 3, 3))

bval_true = np.zeros((mask2.shape[0], mask2.shape[1], mask2.shape[2], bt.shape[0]))
bs_true = np.zeros((mask2.shape[0], mask2.shape[1], mask2.shape[2], bt.shape[0]))
bp_true = np.zeros((mask2.shape[0], mask2.shape[1], mask2.shape[2], bt.shape[0]))
bl_true = np.zeros((mask2.shape[0], mask2.shape[1], mask2.shape[2], bt.shape[0]))




eigval, eigvec = get_btensor_eigen(bt)
bval, bs, bp, bl = get_btensor_shape_topgaard_list(eigval)

nonb0mask = (bval > 1e-3).astype(np.bool)

sphericmask = np.logical_and(bval > 1e-3, bs > 0.9)
planarmask = np.logical_and(bval > 1e-3, bp > 0.9)
linearmask = np.logical_and(bval > 1e-3, bl > 0.9)





tot = mask2[:,:,:].sum()
it = 0

begint = time()
for x in range(mask.shape[0]):
	for y in range(mask.shape[1]):
		for z in range(mask.shape[2]):
			if mask2[x,y,z]:
				# grab local GNL tensor
				L = tensors[x,y,z]

				# distort each b-tensor
				tmp_bt = np.zeros_like(bt)
				tmp_bt = predict_B_list_from_L(bt, L)
				bt_true[x,y,z] = tmp_bt

				eigval, eigvec = get_btensor_eigen(tmp_bt)
				tmp_bval, tmp_bs, tmp_bp, tmp_bl = get_btensor_shape_topgaard_list(eigval)
				bval_true[x,y,z] = tmp_bval
				bs_true[x,y,z] = tmp_bs
				bp_true[x,y,z] = tmp_bp
				bl_true[x,y,z] = tmp_bl


				if not it%1000:
					print('{} / {}'.format(it, tot))

				it += 1
endt = time()


print('elapsed time {} s'.format(endt-begint))
print('for {} vox :: {} s per vox avg'.format(tot, (endt-begint)/float(tot)))



# del bt_true
















textfs = 30


fig = pl.figure()

pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.5)




bins_desired = 41
bins_actual = 200

pl.subplot(2,2,1)



bvs = [0.1, 0.5, 1.0, 1.5, 2.0, 4.0]
for ib, bv in enumerate(bvs):
	pl.axvline(bv, ymin=0, ymax=1, alpha=0.75, color='r')
	bmask = np.logical_and(bval > 0.95*bv, bval < 1.05*bv)
	big_vector = bval_true[mask2][:,bmask].ravel()
	# pl.hist(big_vector, bins=bins_desired, alpha=0.5)
	counts, bins_edge = np.histogram(big_vector, bins=bins_desired)
	pl.bar(bins_edge[:-1], counts/float(counts.max()), width=np.mean(bins_edge[1:]-bins_edge[:-1]), alpha=0.5)

	# pl.text(bv+0.02, 0.97, r'{} $\frac{{m\text{{s}}}}{{\mu\text{{m}}^2}}$'.format(bv), size=textfs, color='r', weight='bold')
	pl.text(bv+0.03, 0.93, '{}'.format(bv), size=textfs-2, color='r', weight='bold')

# pl.legend()
pl.title('b-values for all B-tensors', fontsize=textfs+2)

pl.xticks(fontsize=textfs-4)
pl.xlabel(r'b-value ($m\text{{s}}/\mu\text{{m}}^2$)', fontsize=textfs)
pl.yticks([], [])
pl.ylabel('Proportion', fontsize=textfs)


pl.subplot(2,2,2)
pl.locator_params(nbins=6)
# pl.hist(bs[sphericmask], bins=bins_desired, label='desired', alpha=0.5, density=True)
pl.axvline(bs[sphericmask].mean(), ymin=0, ymax=1, alpha=0.75, color='r')
big_vector = bs_true[mask2][:,sphericmask].ravel()
# pl.hist(big_vector, bins=bins_actual, label='actual', alpha=0.5, density=True)
pl.hist(big_vector, bins=bins_actual, alpha=0.5, density=True, color='b')
# pl.legend()
pl.title(r'Shape of spherical B-tensors', fontsize=textfs+2)

pl.xticks(fontsize=textfs-4)
pl.xlabel(r'Normalized $b_{{spherical}}$ ($b_S/b$)', fontsize=textfs)
pl.yticks([], [])
pl.ylabel('Proportion', fontsize=textfs)

line_patch = mpatches.Patch(color='r', alpha=0.75, label='desired B-tensor')
hist_patch = mpatches.Patch(color='b', alpha=0.5, label='actual B-tensor')
pl.legend(handles=[line_patch, hist_patch], loc=2, ncol=1, fontsize=textfs-2)


pl.subplot(2,2,3)
pl.locator_params(nbins=6)
# pl.hist(bp[planarmask], bins=bins_desired, label='desired', alpha=0.5, density=True)
pl.axvline(bp[planarmask].mean(), ymin=0, ymax=1, alpha=0.75, color='r')
big_vector = bp_true[mask2][:,planarmask].ravel()
# pl.hist(big_vector, bins=bins_actual, label='actual', alpha=0.5, density=True)
pl.hist(big_vector, bins=bins_actual, alpha=0.5, density=True, color='b')
# pl.legend()
pl.title(r'Shape of planar B-tensors', fontsize=textfs+2)

pl.xticks(fontsize=textfs-4)
pl.xlabel(r'Normalized $b_{{planar}}$ ($b_P/b$)', fontsize=textfs)
pl.yticks([], [])
pl.ylabel('Proportion', fontsize=textfs)

line_patch = mpatches.Patch(color='r', alpha=0.75, label='desired B-tensor')
hist_patch = mpatches.Patch(color='b', alpha=0.5, label='actual B-tensor')
pl.legend(handles=[line_patch, hist_patch], loc=2, ncol=1, fontsize=textfs-2)


pl.subplot(2,2,4)
pl.ticklabel_format(useOffset=False)
pl.locator_params(nbins=3)
# pl.hist(bl[linearmask], bins=bins_desired, label='desired', alpha=0.5, density=True)
pl.axvline(bl[linearmask].mean(), ymin=0, ymax=1, alpha=0.75, color='r')
big_vector = bl_true[mask2][:,linearmask].ravel()
# pl.hist(big_vector, bins=bins_actual, label='actual', alpha=0.5, density=True)
pl.hist(big_vector, bins=bins_actual, alpha=0.5, density=True, color='b')
# pl.legend()
pl.title(r'Shape of linear B-tensors', fontsize=textfs+2)

pl.xticks(fontsize=textfs-4)
pl.xlabel(r'Normalized $b_{{linear}}$ ($b_L/b$)', fontsize=textfs)
pl.yticks([], [])
pl.ylabel('Proportion', fontsize=textfs)



span = big_vector.max() - big_vector.min()
pl.xlim(big_vector.min() - 0.1*span, big_vector.max())



line_patch = mpatches.Patch(color='r', alpha=0.75, label='desired B-tensor')
hist_patch = mpatches.Patch(color='b', alpha=0.5, label='actual B-tensor')
pl.legend(handles=[line_patch, hist_patch], loc=2, ncol=1, fontsize=textfs-2)



for n, ax in enumerate(fig.axes):
    ax.text(-0.05, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=textfs+4, weight='bold')


pl.show()



