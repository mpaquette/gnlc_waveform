# Script to generate one of the ISMRM figure
import os

import numpy as np
import pylab as pl
import nibabel as nib

import matplotlib as mpl
# mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

import string

import matplotlib.gridspec as gridspec

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




data_min = norms[mask2].min()
data_max = norms[mask2].max()


# xx,yy,zz = mask2.shape
# cxx = xx//2
# cyy = yy//2
# czz = zz//2


xxx,yyy,zzz = np.where(mask2)
cxx = xxx.min() + (xxx.max() - xxx.min())//2
cyy = yyy.min() + (yyy.max() - yyy.min())//2
czz = zzz.min() + (zzz.max() - zzz.min())//2




cm=pl.cm.inferno


norms = np.ma.array(norms, mask=np.logical_not(mask2))

textfs = 20

fig = pl.figure()



pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.0, hspace=0.1)



# pl.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
pl.subplot(2,2,1)
# pl.imshow((norms*mask2)[cxx,::-1,::-1].T, cmap=cmap, vmin=data_min, vmax=data_max)
# pl.imshow((norms*mask2)[cxx,yyy.max():yyy.min():-1,zzz.max():zzz.min():-1].T, cmap=cmap, vmin=data_min, vmax=data_max)
pl.imshow((norms)[cxx,yyy.max():yyy.min():-1,zzz.max():zzz.min():-1].T, cmap=cm, vmin=data_min, vmax=data_max)
pl.axis('off')
pl.title('Sagittal', fontsize=textfs+2)
pl.gca().set_aspect('equal')
# pl.gca().set_aspect('auto')

# pl.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
pl.subplot(2,2,2)
# pl.imshow((norms*mask2)[::-1,cyy,::-1].T, cmap=cmap, vmin=data_min, vmax=data_max)
# pl.imshow((norms*mask2)[xxx.max():xxx.min():-1,cyy,zzz.max():zzz.min():-1].T, cmap=cmap, vmin=data_min, vmax=data_max)
pl.imshow((norms)[xxx.max():xxx.min():-1,cyy,zzz.max():zzz.min():-1].T, cmap=cm, vmin=data_min, vmax=data_max)
pl.axis('off')
pl.title('Coronal', fontsize=textfs+2)
pl.gca().set_aspect('equal')
# pl.gca().set_aspect('auto')

# pl.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
pl.subplot(2,2,3)
# pl.imshow((norms*mask2)[::-1,::-1,czz].T, cmap=cmap, vmin=data_min, vmax=data_max)
# pl.imshow((norms*mask2)[xxx.max():xxx.min():-1,yyy.max():yyy.min():-1,czz].T, cmap=cmap, vmin=data_min, vmax=data_max)
pl.imshow((norms)[xxx.max():xxx.min():-1,yyy.max():yyy.min():-1,czz].T, cmap=cm, vmin=data_min, vmax=data_max)
pl.axis('off')
pl.title('Axial', fontsize=textfs+2)
pl.gca().set_aspect('equal')
# pl.gca().set_aspect('auto')

# pl.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
pl.subplot(2,2,4)
# pl.colorbar()

# pl.hist(norms[mask2], bins=100)
# pl.hist(norms.ravel(), bins=100)
# pl.axis('off')


Nbins = 100
n, bins, patches = pl.hist(norms.ravel(), bins=Nbins)
bin_color = cm(np.linspace(0,1,len(n)))
for i,c in enumerate(bin_color):
    pl.setp(patches[i], 'facecolor', c)



pl.xticks(fontsize=textfs-4)
pl.xlabel(r'$\text{{GNL}}_{{\text{{str}}}}$', fontsize=textfs)

# pl.tick_params(axis='y', reset=True, which='both', bottom=False, top=False, labelbottom=False)
# pl.tick_params(axis='y', reset=True, which='both', bottom='off', top='off', labelbottom='off')
pl.yticks([], [])
pl.ylabel('Proportion', fontsize=textfs)

# pl.gca().set_aspect('equal')


# for n, ax in enumerate(fig.axes):
#     ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
#             size=textfs+4, weight='bold')



fig.axes[0].text(-0., 1.1, string.ascii_uppercase[0], transform=fig.axes[0].transAxes, size=textfs+4, weight='bold')
# fig.axes[1].text(1.1, 1.1, string.ascii_uppercase[1], transform=fig.axes[0].transAxes, size=textfs+4, weight='bold')
fig.axes[1].text(1.2, 1.1, string.ascii_uppercase[1], transform=fig.axes[0].transAxes, size=textfs+4, weight='bold')
fig.axes[2].text(-0., -0.05, string.ascii_uppercase[2], transform=fig.axes[0].transAxes, size=textfs+4, weight='bold')
# fig.axes[3].text(1.1, -0.1, string.ascii_uppercase[3], transform=fig.axes[0].transAxes, size=textfs+4, weight='bold')
fig.axes[3].text(1.2, -0.05, string.ascii_uppercase[3], transform=fig.axes[0].transAxes, size=textfs+4, weight='bold')






pl.show()


