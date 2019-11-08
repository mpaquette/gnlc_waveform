# Script to generate one of the ISMRM figure
import os

import numpy as np
import pylab as pl

import matplotlib as mpl
# mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


from gnlc_waveform.io_waveform import *
from gnlc_waveform.btensor import *
from gnlc_waveform.viz import *



# simple import and viz with gradient non-linearity distortion
# setup
filenameA = os.path.join(os.path.dirname(__file__), '../waveform/FWF_CUSTOM001_A.txt')
filenameB = os.path.join(os.path.dirname(__file__), '../waveform/FWF_CUSTOM001_B.txt')
GMAX = 169.6e-3 # T/m
DURATION_LEFT = 20.39e-3 # s
DURATION_PAUSE = 8.42e-3 # s
DURATION_RIGHT = 16.51e-3 # s




# read waveform file
gradient_low, t_low, dt_low = read_NOWAB(filenameA, filenameB, DURATION_LEFT, DURATION_PAUSE, DURATION_RIGHT, GMAX, flipB=True)


# resampling low time-resolution gradient to >=1000 points for proper numerical sumation
gradient, t, dt = resample_waveform_equi(gradient_low, t_low, minN=1000)
qt = compute_q_from_G(gradient, dt)

# compute B-tensor and shape
btensor = compute_B_from_q(qt, dt)
eigval, eigvec = get_btensor_eigen(btensor)
bval, bs, bp, bl = get_btensor_shape_topgaard(eigval)


# compute B-tensor and shape
btensor = compute_B_from_q(qt, dt)
eigval, eigvec = get_btensor_eigen(btensor)
bval, bs, bp, bl = get_btensor_shape_topgaard(eigval)


# Introducing an example gradient linear tenser
# print('Introducing distortion with tensor:')
L = np.array([[1.1, 0, 0],[0.01, 1, -0.02],[0, 0, 0.95]])
# print(L)
dist_gradient = distort_G(gradient, L)
gradient_norm = np.linalg.norm(gradient, axis=1)
dist_gradient_norm = np.linalg.norm(dist_gradient, axis=1)



dist_qt = compute_q_from_G(dist_gradient, dt)
# compute B-tensor and shape
dist_btensor = compute_B_from_q(dist_qt, dt)
dist_eigval, dist_eigvec = get_btensor_eigen(dist_btensor)
dist_bval, dist_bs, dist_bp, dist_bl = get_btensor_shape_topgaard(dist_eigval)







pl.figure()
# pl.figure(constrained_layout=True)
gridspec.GridSpec(3,2)



pl.subplots_adjust(left=0.07, bottom=0.07, right=0.9, top=0.95, wspace=0.1, hspace=0.3)


textfs = 28
mathfs = 28

pl.subplot2grid((2,3), (0,0), colspan=2, rowspan=1)

pl.plot(t, gradient[:,0], label=r'$X_{desired}$', color='r')
pl.plot(t, gradient[:,1], label=r'$Y_{desired}$', color='g')
pl.plot(t, gradient[:,2], label=r'$Z_{desired}$', color='b')
pl.plot(t, dist_gradient[:,0], label=r'$X_{actual}$', color='r', linestyle='--')
pl.plot(t, dist_gradient[:,1], label=r'$Y_{actual}$', color='g', linestyle='--')
pl.plot(t, dist_gradient[:,2], label=r'$Z_{actual}$', color='b', linestyle='--')

pl.xlabel(r'time ($s$)', fontsize=textfs)
pl.ylabel(r'gradient strength ($T m^{-1}$)', fontsize=textfs)
pl.title('Gradient Waveform', fontsize=textfs+2)
pl.xlim(0,t.max())
# pl.legend()

x_patch = mpatches.Patch(color='red', label='X')
y_patch = mpatches.Patch(color='green', label='Y')
z_patch = mpatches.Patch(color='blue', label='Z')

fake_fullline = pl.Line2D([],[], label='desired', color='black', linestyle='-')
fake_dashedline = pl.Line2D([],[], label='actual', color='black', linestyle='--')

pl.legend(handles=[x_patch, y_patch, z_patch, fake_fullline, fake_dashedline], loc=8, ncol=2, fontsize=textfs-2)



pl.subplot2grid((2,3), (1,0), colspan=2, rowspan=1)

pl.plot(t, qt[:,0], label=r'$X_{desired}$', color='r')
pl.plot(t, qt[:,1], label=r'$Y_{desired}$', color='g')
pl.plot(t, qt[:,2], label=r'$Z_{desired}$', color='b')
pl.plot(t, dist_qt[:,0], label=r'$X_{actual}$', color='r', linestyle='--')
pl.plot(t, dist_qt[:,1], label=r'$Y_{actual}$', color='g', linestyle='--')
pl.plot(t, dist_qt[:,2], label=r'$Z_{actual}$', color='b', linestyle='--')
pl.xlabel(r'time ($s$)', fontsize=textfs)
pl.ylabel(r'q ($m^{-1}$)', fontsize=textfs)
pl.title('q-Vector', fontsize=textfs+2)
pl.xlim(0,t.max())
# pl.legend()

x_patch = mpatches.Patch(color='red', label='X')
y_patch = mpatches.Patch(color='green', label='Y')
z_patch = mpatches.Patch(color='blue', label='Z')

fake_fullline = pl.Line2D([],[], label='desired', color='black', linestyle='-')
fake_dashedline = pl.Line2D([],[], label='actual', color='black', linestyle='--')

pl.legend(handles=[x_patch, y_patch, z_patch, fake_fullline, fake_dashedline], loc=8, ncol=2, fontsize=textfs-2)


pl.subplot2grid((2,3), (0,2))
pl.axis('off')
pl.xlim(0,1)
pl.ylim(0,1)
pl.text(0.0, 0.6, r"$\mathbf{{L}} = \begin{{bmatrix}} {} & {} & {} \\ {} & {} & {} \\ {} & {} & {} \end{{bmatrix}}$".format(*L.ravel()), fontsize=mathfs, color='black')
pl.text(0.0, 0.4, r"$\text{{GNL}}_{{\text{{str}}}} = \| \mathbf{{L}} - \mathbf{{I}}_3 \|_F = {:.3f} $".format(np.linalg.norm(L-np.eye(3))), fontsize=mathfs, color='black')

pl.subplot2grid((2,3), (1,2))
pl.axis('off')
pl.xlim(0,1)
pl.ylim(0,1)
pl.text(0.0, 0.85, r"$\mathbf{{B}} = \begin{{bmatrix}} {:.2f} & {:.2f} & {:.2f} \\ {:.2f} & {:.2f} & {:.2f} \\ {:.2f} & {:.2f} & {:.2f} \end{{bmatrix}} \, \frac{{m\text{{s}}}}{{\mu\text{{m}}^{{2}}}}$".format(*(1e-9*btensor.ravel())), fontsize=mathfs, color='black')
pl.text(0.0, 0.73, r"$b = {:.2f} \,m\text{{s}}\,\mu\text{{m}}^{{-2}}$".format(1e-9*bval), fontsize=mathfs, color='black')
pl.text(0.0, 0.63, r"$b_{{S}}/b = {:.2f}$".format(bs), fontsize=mathfs, color='black')


pl.text(0.0, 0.15, r"$\mathbf{{B}}_a = \begin{{bmatrix}} {:.2f} & {:.2f} & {:.2f} \\ {:.2f} & {:.2f} & {:.2f} \\ {:.2f} & {:.2f} & {:.2f} \end{{bmatrix}} \, \frac{{m\text{{s}}}}{{\mu\text{{m}}^{{2}}}}$".format(*(1e-9*dist_btensor.ravel())), fontsize=mathfs, color='black')
pl.text(0.0, 0.03, r"$b = {:.2f}\,m\text{{s}}\,\mu\text{{m}}^{{-2}}$".format(1e-9*dist_bval), fontsize=mathfs, color='black')
pl.text(0.0, -0.07, r"$b_{{S}}/b = {:.2f}$".format(dist_bs), fontsize=mathfs, color='black')


pl.show()



