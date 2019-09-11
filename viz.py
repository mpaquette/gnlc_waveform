import numpy as np
import pylab as pl
from scipy.ndimage import center_of_mass

def peraxisplot(x, ydata, title='', xlabel='', ylabel='', axhline=[], axvline=[], extra_label=''):
	# plot Nx3 vector data as 3 different curves
	pl.plot(x, ydata[:,0], label=extra_label+'X')
	pl.plot(x, ydata[:,1], label=extra_label+'Y')
	pl.plot(x, ydata[:,2], label=extra_label+'Z')
	pl.xlabel(xlabel)
	pl.ylabel(ylabel)
	pl.legend()
	for xx in axhline:
		pl.axhline(xx, color='k')
	for xx in axvline:
		pl.axvline(xx, color='k')
	pl.title(title)


def plot(x, ydata, label='', title='', xlabel='', ylabel='', axhline=[], axvline=[]):
	# plot Nx1 data
	pl.plot(x, ydata, label=label)
	pl.xlabel(xlabel)
	pl.ylabel(ylabel)
	if label != '':
		pl.legend()
	for xx in axhline:
		pl.axhline(xx, color='k')
	for xx in axvline:
		pl.axvline(xx, color='k')
	pl.title(title)


def xyzplot(data, mask=None, title='', distval=''):
	# plot X x Y x Z data as 3 slice and an histogram
	# default mask to full image
	if mask is None:
		mask = np.ones_like(data)
	mask = mask.astype(np.bool)
	# get mask center of mask to automatically choose slices
	cx, cy, cz = center_of_mass(mask)
	cx = int(np.round(cx))
	cy = int(np.round(cy))
	cz = int(np.round(cz))

	pl.figure(num=title)

	pl.subplot(2,2,1)
	pl.imshow(np.ma.masked_where(np.logical_not(mask), data)[:,::-1,cz].T)
	# pl.imshow((data*mask)[:,::-1,cz].T)
	pl.title('X-Y (Z = {})'.format(cz))
	pl.axis('off')

	pl.subplot(2,2,2)
	pl.imshow(np.ma.masked_where(np.logical_not(mask), data)[:,cy,::-1].T)
	# pl.imshow((data*mask)[:,cy,::-1].T)
	pl.title('X-Z (Y = {})'.format(cy))
	pl.axis('off')

	pl.subplot(2,2,3)
	pl.imshow(np.ma.masked_where(np.logical_not(mask), data)[cx,:,::-1].T)
	# pl.imshow((data*mask)[cx,:,::-1].T)
	pl.title('Y-Z (X = {})'.format(cx))
	pl.axis('off')

	pl.subplot(2,2,4)
	pl.hist((data[mask].ravel()), bins=100, color='b')
	pl.axvline(distval, color='r')
	pl.title('Distribution (GT = {})'.format(distval))

	# pl.show()



