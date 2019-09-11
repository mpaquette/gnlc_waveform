# import numpy as np
import pylab as pl


def peraxisplot(x, ydata, title='', xlabel='', ylabel='', axhline=[], axvline=[], extra_label=''):
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

