import caffe
import numpy as np


class DistanceLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
		if len(bottom) != 2:
			raise Exception("Need two inputs to compute distance.")

	def reshape(self, bottom, top):
		self.grad = []
		self.diff = np.zeros((bottom[0].data.shape[0], bottom[0].data.shape[1]))
		top[0].reshape(bottom[0].num)  # distance

	def forward(self, bottom, top):
		self.diff = bottom[0].data - bottom[1].data
		top[0].data[...] = np.sum(self.diff ** 2, axis=1)

	def backward(self, top, propagate_down, bottom, gradientChech=False):
		if propagate_down[0]:
			bottom[0].diff[...] = 2. * self.diff * top[0].diff[:, None]

		if propagate_down[1]:
			bottom[1].diff[...] = -2. * self.diff * top[0].diff[:, None]

		if gradientChech:
			epsilon = 1e-4

			self.bPlus = bottom[0].data + epsilon
			self.bMinus = bottom[0].data - epsilon
			self.gradPlus = ((self.bPlus - bottom[1].data) ** 2)
			self.gradMinus = ((self.bMinus - bottom[1].data) ** 2)
			self.grad.append((self.gradPlus - self.gradMinus) / (2. * epsilon))

			self.bPlus = bottom[1].data + epsilon
			self.bMinus = bottom[1].data - epsilon
			self.gradPlus = ((bottom[0].data - self.bPlus) ** 2)
			self.gradMinus = ((bottom[0].data - self.bMinus) ** 2)
			self.grad.append((self.gradPlus - self.gradMinus) / (2. * epsilon))

			out = self.grad[0] * top[0].diff[:, None]
			print out[0]
			print bottom[0].diff[0]
			out = self.grad[1] * top[0].diff[:, None]
			print out[0]
			print bottom[1].diff[0]
			print('\n')
