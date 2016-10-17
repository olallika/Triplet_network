import caffe
import numpy as np


class L2NormLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
		if len(bottom) != 1:
			raise Exception("Need one input to compute distance.")

	def reshape(self, bottom, top):
		self.sum = np.zeros((bottom[0].data.shape[0], 1))
		top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1])

	def forward(self, bottom, top):
		self.sum = np.sqrt(np.sum(bottom[0].data ** 2, axis=1))
		top[0].data[...] = bottom[0].data / self.sum[:, None]

	def backward(self, top, propagate_down, bottom, gradientChech=False):
		if propagate_down[0]:
			norms = self.sum.reshape((bottom[0].num, 1))
			aux = self.sum ** 3
			bottom[0].diff[...] = ((1. / np.tile(norms, (1, bottom[0].data.shape[1]))) - (
			(bottom[0].data ** 2) / aux[:, None])) * top[0].diff

		if gradientChech:
			epsilon = 1e-4
			aux = np.sqrt(np.sum((bottom[0].data + epsilon) ** 2, axis=1))
			gradPlus = (bottom[0].data + epsilon) / aux[:, None]
			aux = np.sqrt(np.sum((bottom[0].data - epsilon) ** 2, axis=1))
			gradMinus = (bottom[0].data - epsilon) / aux[:, None]
			grad = (gradPlus - gradMinus) / (2. * epsilon)

			print grad * top[0].diff
			print bottom[0].diff
			print('\n')
