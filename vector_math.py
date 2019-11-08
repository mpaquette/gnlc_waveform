import numpy as np


# vector projection
def proj(u,v):
	return (np.dot(u,v)/np.dot(u,u)) * u


# angle between 3d vectors
def angleVec3(v1, v2):
	ang = np.math.acos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1))
	return ang


# checks if 3 3d vectors are orthogonal
def checkOrtho3(u1, u2, u3, tol=1e-3):
	a12 = angleVec3(u1, u2)
	a13 = angleVec3(u1, u3)
	a23 = angleVec3(u2, u3)
	if (np.abs(a12-np.pi/2.) < tol) and (np.abs(a13-np.pi/2.) < tol) and (np.abs(a23-np.pi/2.) < tol):
		return True
	else:
		return False


# outputs 3 linearly independant 3d vector from a starting one
def findLinIndepRandomRot(v):
	# Assumes the input has norm > 0
	v1 = v / np.linalg.norm(v)
	idx = np.argsort(v1)
	ang = 2*np.pi*np.random.rand()
	v2 = np.zeros(3)
	v2[idx[0]] = np.cos(ang)
	v2[idx[1]] = np.sin(ang)
	v3 = np.zeros(3)
	v3[idx[0]] = -np.sin(ang)
	v3[idx[1]] = np.cos(ang)
	return v1, v2, v3


# produce 3 orthonormal 3d vectors from 3 linearly independant 3d vector
def gramSchmidt3(v1, v2, v3):
	# produce vectors u1 u2 u3 which are orthogonal and e1 e2 e3 which are orthonormal
	# all vector are normalized
	# need v1 v2 v3 to be linearly independant (use findLinIndep())
	# u1 has the same orienation as v1
	# Gram-Schmidt 3 vector
	u1 = v1.copy()
	u2 = v2 - proj(u1,v2)
	u3 = v3 - proj(u1,v3) - proj(u2,v3)
	# e1,e2,e3 are orthonormal
	# e1 is parallel to dir therefore e2,e3 span a perpendicular plane to dir 
	e1 = u1 / np.linalg.norm(u1)
	e2 = u2 / np.linalg.norm(u2)
	e3 = u3 / np.linalg.norm(u3)
	return e1, e2, e3


# generate axisymmetric tensor from 3 eigenvectors and 2 eigenvalues
def axisymTensor(e1, e2, e3, lpara, lperp, verbose=0):
	# generate axisymmetric tensor with the following eigen vector/value pair:
	# e1 lpara
	# e2 lperp
	# e3 lperp
	if not checkOrtho3(e1, e2, e3):
		print('Basis doesnt seems orthogonal, quiting')
		return None

	n1 = np.linalg.norm(e1)
	n2 = np.linalg.norm(e2)
	n3 = np.linalg.norm(e3)
	if verbose:
		print('Basis norms (should be 1):')
		print(n1)
		print(n2)
		print(n3)

	return lpara * np.outer(e1,e1) + lperp * np.outer(e2,e2) + lperp * np.outer(e3,e3)


def give_tensor_bval(T, b):
	# make tensor T have bvalue b
	# bval(T) = trace(T)
	return (b/np.trace(T)) * T


# bad random tensor generation
def random_axi_tensor(N=1, minD=0.1, maxD=3):
	# generate N axisymmetric tensor matrix with highest eigenvalue aling in Z
	tmp = minD + (maxD-minD)*np.random.rand(2,N)
	D = np.zeros((N, 3, 3))
	D[:,0,0] = np.min(tmp, axis=0)
	D[:,1,1] = np.min(tmp, axis=0)
	D[:,2,2] = np.max(tmp, axis=0)
	return D

