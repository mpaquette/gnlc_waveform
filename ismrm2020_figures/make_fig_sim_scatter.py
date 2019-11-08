# Script to generate one of the ISMRM figure
import numpy as np
import pylab as pl
import nibabel as nib

import os

# from gnlc_waveform.vector_math import 
from gnlc_waveform.tensor_math import _S
from gnlc_waveform.dtd_cov import dtd_cov_1d_data2fit, convert_m

from time import time

import matplotlib as mpl
# mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}\usepackage{{amsfonts}}'

import string
# import matplotlib.patches as mpatches

from gnlc_waveform.btensor import predict_B_list_from_L

from dipy.data import get_sphere





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




### remove b = 4
# bt = bt[np.trace(bt, axis1=-2, axis2=-1) < 2.5]




L = np.load(os.path.join(os.path.dirname(__file__), '../gnlt/dev_parc_centroid_n_100.npy')).reshape((-1,3,3))
norms = np.linalg.norm(L-np.eye(3), axis=(-2,-1))

tmp_bt = [predict_B_list_from_L(bt, L[i]) for i in range(L.shape[0])]




sphere = get_sphere('repulsion724').subdivide(1)



mu = np.array([1.0, 1.0, 1.0])
mu = mu/np.linalg.norm(mu)
angles = np.arccos(np.dot(sphere.vertices, mu))*(180/np.pi)


angs = [15]
dist = []

propers_d = []
propers_c = []
wrongs_d = []
wrongs_c = []
for ang_th in angs:
	clip = angles < ang_th


	from planar.gs import findLinIndepRandomRot, gramSchmidt3
	def gentensor(e1, dpar, dperp):
		v1, v2, v3 = findLinIndepRandomRot(e1)
		u1, u2, u3 = gramSchmidt3(v1, v2, v3)
		return dpar*np.outer(u1, u1) + dperp*np.outer(u2,u2) + dperp*np.outer(u3,u3)

	# diagonalized diffusivities
	lam_para = 1.5
	lam_perp = 0.3



	D = []
	for e1 in sphere.vertices[clip]:
		D.append(gentensor(e1, lam_para, lam_perp))

	D = np.array(D)
	N = D.shape[0]


	# vol fracs
	PD = np.ones(N)
	PD /= PD.sum()



	# generate signal from D distribution
	S_desired = np.array([PD[i] * _S(bt, D[i]) for i in range(D.shape[0])]).sum(axis=0)
	S_actual = [np.array([PD[i] * _S(tmp_bt[j], D[i]) for i in range(D.shape[0])]).sum(axis=0) for j in range(len(tmp_bt))]


	m = dtd_cov_1d_data2fit(S_desired, bt, cond_limit=1e-10)
	s0, d, c = convert_m(m)
	proper_d = []
	proper_c = []
	wrong_d = []
	wrong_c = []
	for j in range(len(tmp_bt)):
		m_fit = dtd_cov_1d_data2fit(S_actual[j], tmp_bt[j], cond_limit=1e-10)
		s0_fit, d_fit, c_fit = convert_m(m_fit)
		proper_d.append(np.linalg.norm(d-d_fit))
		proper_c.append(np.linalg.norm(c-c_fit))
		propers_d.append(proper_d)
		propers_c.append(proper_c)

		m_fit = dtd_cov_1d_data2fit(S_actual[j], bt, cond_limit=1e-10)
		s0_fit, d_fit, c_fit = convert_m(m_fit)
		wrong_d.append(np.linalg.norm(d-d_fit))
		wrong_c.append(np.linalg.norm(c-c_fit))
		wrongs_d.append(wrong_d)
		wrongs_c.append(wrong_c)




i=0


textfs = 32
markersize = 200

fig = pl.figure()

pl.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.88, wspace=0.35, hspace=0.1)



pl.subplot(1,2,1)
pl.locator_params(nbins=7)
# pl.scatter(norms, wrongs_d[i], alpha=0.75, s=markersize, label='desired B-tensors')
pl.scatter(norms, wrongs_d[i], alpha=0.75, s=markersize, label=r'Fit using $\mathbf{{B}}$', edgecolors=None)
# pl.scatter(norms, propers_d[i], alpha=0.75, s=markersize, label='actual B-tensors')
pl.scatter(norms, propers_d[i], alpha=0.75, s=markersize, label=r'Fit using $\mathbf{{B}}_a$', edgecolors=None)
k = max(max(propers_d[i]), max(wrongs_d[i]))
pl.ylim(-0.03*k, 1.01*k)
pl.xlim(0, norms.max()*1.01)
pl.title(r'Difference on $\langle\mathbf{{D}}\rangle$ with GNL correction', fontsize=textfs+2)
pl.xlabel(r'$\text{{GNL}}_{{\text{{str}}}}$', fontsize=textfs)
pl.ylabel(r'$\| \langle\mathbf{{D}}\rangle_{\text{no-GNL}} - \langle\mathbf{{D}}\rangle_{\text{GNL}} \|_F$', fontsize=textfs)
pl.legend(fontsize=textfs)
pl.xticks(fontsize=textfs-4)
pl.yticks(fontsize=textfs-4)

pl.subplot(1,2,2)
pl.locator_params(nbins=7)
# pl.scatter(norms, wrongs_c[i], alpha=0.75, s=markersize, label='desired B-tensors')
pl.scatter(norms, wrongs_c[i], alpha=0.75, s=markersize, label=r'Fit using $\mathbf{{B}}$', edgecolors=None)
# pl.scatter(norms, propers_c[i], alpha=0.75, s=markersize, label='actual B-tensors')
pl.scatter(norms, propers_c[i], alpha=0.75, s=markersize, label=r'Fit using $\mathbf{{B}}_a$', edgecolors=None)
k = max(max(propers_c[i]), max(wrongs_c[i]))
pl.ylim(-0.03*k, 1.01*k)
pl.xlim(0, norms.max()*1.01)
pl.title(r'Difference on $\mathbb{{C}}$ with GNL correction', fontsize=textfs+2)
pl.xlabel(r'$\text{{GNL}}_{{\text{{str}}}}$', fontsize=textfs)
pl.ylabel(r'$\| \mathbb{{C}}_{\text{no-GNL}} - \mathbb{{C}}_{\text{GNL}} \|_F$', fontsize=textfs)
pl.legend(fontsize=textfs)
pl.xticks(fontsize=textfs-4)
pl.yticks(fontsize=textfs-4)





for n, ax in enumerate(fig.axes):
    ax.text(-0.15, 1.07, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=textfs+4, weight='bold')



pl.show()
