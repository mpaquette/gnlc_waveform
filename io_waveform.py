import numpy as np 
from scipy.interpolate import interp1d


def read_topgaard(filename, GMAX):
	# load gradient waveform from :filename: generated with
	# https://github.com/daniel-topgaard/md-dmri/blob/master/acq/bruker/paravision/make_waveform.m
	# :GMAX: is the maximum gradient strength in T/m

	# read textfile 
	f = open(filename,'r')
	lines = f.readlines()
	f.close()

	# read relevant header lines
	BFACTOR = float(lines[16].strip().split('=')[-1])
	DURATION = float(lines[17].strip().split('=')[-1])
	DIRECTIONVEC = np.array(lines[18].split('=')[-1].strip().split(' '), dtype=np.float)
	NPOINTS = int(lines[19].strip().split('=')[-1])

	# compute time duration of 1 interval between 2 points (constant)
	dt = DURATION / float(NPOINTS-1) # s 

	# read normalized gradient waveform
	# skip header
	vector = np.genfromtxt(filename, skip_header=21)

	# Rescale gradient to T/m
	vector *= GMAX

	# help detect wrong units or badly normalized waveform
	print('axis Gmax is {} T/m'.format(np.abs(vector).max()))
	print('norm Gmax is {} T/m'.format(np.linalg.norm(vector, axis=1).max()))

	# time discretization
	t = np.linspace(0, DURATION, NPOINTS)

	return vector, t, dt



def read_NOW(filename, DURATION_LEFT, DURATION_PAUSE, DURATION_RIGHT, GMAX):
	# load gradient waveform from :filename: generated with
	# https://github.com/jsjol/NOW
	# :GMAX: is the maximum gradient strength in T/m
	# :DURATION_LEFT: is the left waveform duration in s
	# :DURATION_PAUSE: is the pause between waveform duration in s
	# :DURATION_RIGHT: is the right waveform duration in s

	# read normalized gradient waveform
	# skip first line containing
	vector = np.genfromtxt(filename, skip_header=1)
	NPOINTS = vector.shape[0]

	# compute time duration of 1 interval between 2 points (constant)
	DURATION = DURATION_LEFT + DURATION_PAUSE + DURATION_RIGHT
	dt = DURATION / float(NPOINTS-1) # s 

	# Rescale gradient to T/m
	# can go up sqrt(3) ~ 1.73 if the waveform was made with MAXNORM
	# should not go much much higher than 1 if the waveform was made with euclidean nrom
	vector *= GMAX
	print('axis Gmax is {} T/m'.format(np.abs(vector).max()))
	print('norm Gmax is {} T/m'.format(np.linalg.norm(vector, axis=1).max()))

	# time discretization
	t = np.linspace(0, DURATION, NPOINTS)

	return vector, t, dt




def resample_waveform_equi(vector, t, minN):
	# linearly resample the waveform while keeping all the original point and keeping the point equispaced
	# :vector: is the gradient waveform
	# :t: is the time discretization
	# :minN: is the minimum number of points in the resampled waveform (in practice it will be a bit more)
	
	NPOINTS = vector.shape[0]

	# Compute required new number of points 
	k = int(np.ceil((minN - NPOINTS) / float(NPOINTS-1)))
	Nnew = NPOINTS + (NPOINTS-1)*k

	# new time discretization
	tnew = np.linspace(t[0], t[-1], Nnew)

	# fit interpolation kernel
	f_x_lin = interp1d(t, vector[:,0])
	f_y_lin = interp1d(t, vector[:,1])
	f_z_lin = interp1d(t, vector[:,2])

	# sample interpolator at new time discretization
	newVector = np.zeros((len(tnew),3))
	newVector[:,0] = f_x_lin(tnew)
	newVector[:,1] = f_y_lin(tnew)
	newVector[:,2] = f_z_lin(tnew)

	# it's constant by construction
	dtnew = (tnew[1:] - tnew[:-1]).mean()

	return newVector, tnew, dtnew


def read_NOWAB(filenameA, filenameB, DURATION_LEFT, DURATION_PAUSE, DURATION_RIGHT, GMAX, flipB=False):
	# load gradient waveform from :filename: generated with
	# https://github.com/jsjol/NOW
	# :GMAX: is the maximum gradient strength in T/m
	# :DURATION_LEFT: is the left waveform duration in s
	# :DURATION_PAUSE: is the pause between waveform duration in s
	# :DURATION_RIGHT: is the right waveform duration in s

	# read normalized gradient waveform
	# skip first line containing
	vectorA = np.genfromtxt(filenameA, skip_header=1)
	NPOINTSA = vectorA.shape[0]
	vectorB = np.genfromtxt(filenameB, skip_header=1)
	NPOINTSB = vectorB.shape[0]
    if flipB:
    	vectorB *= -1

    # compute time duration of 1 interval between 2 points (constant)
	dtA = DURATION_LEFT / float(NPOINTSA-1) # s 
	dtB = DURATION_RIGHT / float(NPOINTSB-1) # s 
	dt = (dtA + dtB) / 2
    
	NPOINTS0 = int(round(DURATION_PAUSE / float(dt))) - 1
	vector0 = np.zeros((NPOINTS0, 3))
    
	vector = np.concatenate((vectorA,vector0,vectorB), axis=0)
    
	DURATION = DURATION_LEFT + DURATION_PAUSE + DURATION_RIGHT
	NPOINTS = NPOINTSA + NPOINTS0 + NPOINTSB

	# Rescale gradient to T/m
	# can go up sqrt(3) ~ 1.73 if the waveform was made with MAXNORM
	# should not go much much higher than 1 if the waveform was made with euclidean nrom
	vector *= GMAX
	print('axis Gmax is {} T/m'.format(np.abs(vector).max()))
	print('norm Gmax is {} T/m'.format(np.linalg.norm(vector, axis=1).max()))

	# time discretization
	t = np.linspace(0, DURATION, NPOINTS)

	return vector, t, dt




