import numpy as np


GYRO = 42.57747892e6 * 2*np.pi # rad/s T


def distort_G(gradient, gnl_tensor):
	# Distort the gradient waveform with the gradient non-linearity
	# :gradient: is the gradient waveform (T/m)
	# :gnl_tensor: is the gradient non-linearity tensor
	return gnl_tensor.dot(gradient.T).T

def predict_B_from_L(B, L):
	# predict the distorted B-tensor from the undistorted one and the GNL tensor
	dist_B = np.zeros((3,3))
	# Bxx
	dist_B[0,0] = L[0,0]**2*B[0,0] + 2*L[0,0]*L[0,1]*B[0,1] + 2*L[0,0]*L[0,2]*B[0,2] + L[0,1]**2*B[1,1] + 2*L[0,1]*L[0,2]*B[1,2] + L[0,2]**2*B[2,2]
	# Bxy
	dist_B[0,1] = L[0,0]*L[1,0]*B[0,0] + (L[0,0]*L[1,1] + L[0,1]*L[1,0])*B[0,1] + (L[0,0]*L[1,2] + L[0,2]*L[1,0])*B[0,2] + L[0,1]*L[1,1]*B[1,1] + (L[0,1]*L[1,2] + L[0,2]*L[1,1])*B[1,2] + L[0,2]*L[1,2]*B[2,2]
	dist_B[1,0] = dist_B[0,1]
	# Bxz
	dist_B[0,2] = L[0,0]*L[2,0]*B[0,0] + (L[0,0]*L[2,1] + L[0,1]*L[2,0])*B[0,1] + (L[0,0]*L[2,2] + L[0,2]*L[2,0])*B[0,2] + L[0,1]*L[2,1]*B[1,1] + (L[0,1]*L[2,2] + L[0,2]*L[2,1])*B[1,2] + L[0,2]*L[2,2]*B[2,2]
	dist_B[2,0]
	# Byy
	dist_B[1,1] = L[1,0]**2*B[0,0] + 2*L[1,0]*L[1,1]*B[0,1] + 2*L[1,0]*L[1,2]*B[0,2] + L[1,1]**2*B[1,1] + 2*L[1,1]*L[1,2]*B[1,2] + L[1,2]**2*B[2,2]
	# Byz
	dist_B[1,2] = L[1,0]*L[2,0]*B[0,0] + (L[1,0]*L[2,1] + L[1,1]*L[2,0])*B[0,1] + (L[1,0]*L[2,2] + L[1,2]*L[2,0])*B[0,2] + L[1,1]*L[2,1]*B[1,1] + (L[1,1]*L[2,2] + L[1,2]*L[2,1])*B[1,2] + L[1,2]*L[2,2]*B[2,2]
	dist_B[2,1] = dist_B[1,2]
	# Bzz
	dist_B[2,2] = L[2,0]**2*B[0,0] + 2*L[2,0]*L[2,1]*B[0,1] + 2*L[2,0]*L[0,2]*B[0,2] + L[2,1]**2*B[1,1] + 2*L[2,1]*L[2,2]*B[1,2] + L[2,2]**2*B[2,2]
	return dist_B




def compute_q_from_G(gradient, dt):
	# compute the q-space vectors from the gradient vectors
	# :gradient: is the gradient waveform (T/m)
	# :dt: is the (assumed constant) time interval between 2 points (s)
	#
	# q(t) = GYRO integral_[0,t] g(t) dt
	# in m^-1
	return GYRO * np.cumsum(gradient*dt, axis=0)


def split_q(qt):
	# compute the q-space vectors norms and orientation from the q-space vectors 
	# :qt: is the q-vectors
	qt_norm = np.linalg.norm(qt, axis=1)
	qt_ori = qt / qt_norm[:, None]
	qt_ori[np.isnan(qt_ori)] = 0
	# return norms and unit-norm orientations
	return qt_norm, qt_ori


def compute_B_from_q(qt, dt):
	# compute B-tensor from the q-space vectors 
	# :qt: is the q-vectors (m^-1)
	# :dt: is the (assumed constant) time interval between 2 points (s)
	qt_norm, qt_ori = split_q(qt)
	qt_outer = np.array([np.outer(qt_ori[i,:], qt_ori[i,:]) for i in range(qt_ori.shape[0])])
	btensor = (dt*qt_norm[:,None, None]**2 * qt_outer).sum(0) # (s/m^2)
	return btensor


def get_btensor_eigen(btensor):
	# compute the eigenvalue and eigenvectors of the btensor
	# :btensor: (s/m^2)
	eigval, eigvec = np.linalg.eig(btensor)
	return eigval, eigvec


def get_btensor_shape_topgaard(eigenval):
	# compute the Topgaard btensor shape (spheric, planar, linear)
	eig = np.sort(eigenval) # small to big
	bs = 3*eig[0] # spheric
	bp = 2 * (eig[1] - (bs/3.)) # planar
	bl = eig[2] - (bs/3.) - (bp/2.) # linear
	bval = np.sum(eig)
	# normalize shape
	return bval, bs/float(bval), bp/float(bval), bl/float(bval)



