import numpy as np
import nibabel as nib


def fsl_2_tensor(dev_x_path, dev_y_path, dev_z_path):
	# build voxelwise gradient non-linearity tensor from the output of FSL's calc_grad_perc_dev (applied to the ouput of grad_unwarp.py (https://github.com/mpaquette/gradunwarp))
	dev_X = nib.load(dev_x_path).get_data()
	dev_Y = nib.load(dev_y_path).get_data()
	dev_Z = nib.load(dev_z_path).get_data()
	# convert back from "percentage deviation"
	dev_X *= 0.01
	dev_Y *= 0.01
	dev_Z *= 0.01
	# concatenate into 5D volume
	tensors = np.concatenate((dev_X[...,None], dev_Y[...,None], dev_Z[...,None]), axis=4)
	return tensors


def compute_gnl_score(tensors):
	# compute the Gradient Non-Linearity score of the gnl tensor
	# The score is pseudo distance between GNL tensor and the identity matrix
	# This score penalizes positives and negatives gradient deviations equally, does not prioritize any axis and captures information from the cross-terms.
	# score = l2-norm(eigval(tensor) - I)
	_,s,_ = np.linalg.svd(tensors)
	dist_score = np.linalg.norm(s-np.ones(3), axis=-1)
	return dist_score

