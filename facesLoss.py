import caffe
import numpy as np
import yaml


class FacesLossLayer(caffe.Layer):
	def setup(self, bottom, top):
		# check input pair
		if len(bottom) != 2:
			raise Exception("Need two inputs to compute distance.")
		self.num = yaml.load(self.param_str)['num']
		print "Parameter alpha : ", self.num

	def reshape(self, bottom, top):
		self.diff = np.zeros(bottom[0].data.shape[0])
		self.indexes = np.zeros(bottom[0].data.shape[0])
		self.grad = []
		self.alpha = self.num
		top[0].reshape(1)

	def forward(self, bottom, top):
		self.diff[...] = bottom[0].data - bottom[1].data + self.alpha
		self.indexes = (self.diff > 0)
		top[0].data[...] = np.sum(self.diff[self.indexes])

	def backward(self, top, propagate_down, bottom, gradientCheck=False):
		if propagate_down[0]:
			bottom[0].diff[...] = 1
			bottom[0].diff[np.invert(self.indexes)] = 0

		if propagate_down[1]:
			bottom[1].diff[...] = -1
			bottom[1].diff[np.invert(self.indexes)] = 0

		if gradientCheck:
			epsilon = 1e-4

			self.bPlus = (bottom[0].data + epsilon) - bottom[1].data + self.alpha
			self.gradPlus = np.zeros_like(self.bPlus)
			indexes = self.bPlus > 0
			self.gradPlus = self.bPlus
			self.gradPlus[np.invert(indexes)] = 0
			self.bMinus = (bottom[0].data - epsilon) - bottom[1].data + self.alpha
			self.gradMinus = np.zeros_like(self.bMinus)
			indexes = self.bMinus > 0
			self.gradMinus = self.bMinus
			self.gradMinus[np.invert(indexes)] = 0
			self.grad.append((self.gradPlus - self.gradMinus) / (2. * epsilon))

			self.bPlus = bottom[0].data - (bottom[1].data + epsilon) + self.alpha
			indexes = self.bPlus > 0
			self.gradPlus = self.bPlus
			self.gradPlus[np.invert(indexes)] = 0
			self.bMinus = bottom[0].data - (bottom[1].data - epsilon) + self.alpha
			indexes = self.bMinus > 0
			self.gradMinus = self.bMinus
			self.gradMinus[np.invert(indexes)] = 0
			self.grad.append((self.gradPlus - self.gradMinus) / (2. * epsilon))

			print self.grad[0]
			print bottom[0].diff
			print self.grad[1]
			print bottom[1].diff
			print('\n')
