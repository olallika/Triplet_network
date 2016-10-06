from random import choice
import os
import struct
from array import array as pyarray
import cPickle
from pylab import *
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def printInfo(solver):
	# each output is (batch size, feature dim, spatial dim)
	print '\n >> BLOBS: '
	for k, v in solver.net.blobs.items():
		print k, v.data.shape

	# just print the weight sizes (we'll omit the biases)
	print '\n >> PARAMS (weight sizes): '
	for k, v in solver.net.params.items():
		print k, v[0].data.shape

	# each output is (batch size, feature dim, spatial dim)
	print '\n >> BLOBS (in test): '
	for k, v in solver.test_nets[0].blobs.items():
		print k, v.data.shape

	# just print the weight sizes (we'll omit the biases)
	print '\n >> PARAMS (weight sizes in test): '
	for k, v in solver.test_nets[0].params.items():
		print k, v[0].data.shape



	# Before taking off, let's check that everything is loaded as we expect.
	# We'll run a forward pass on the train and test nets and check that they contain our data.
	# solver.net.forward()  # train net
	# solver.test_nets[0].forward()  # test net (there can be more than one)

	# # We take the train batch size. Get all pairs of images from the first batch and save them together within a folder 'data/image_pairs/'
	# # In this folder we distiguish among pairs of the same class and pairs of different classes
	# # Note that class=1 means being from the same class (while in LeCun's original paper is the other way around!)
	# train_batch_size = solver.net.blobs['pair_data'].data.shape[0]
	# for i in arange(train_batch_size):
	#
	#     # Get the first pair of images and its label
	#     im1   = solver.net.blobs['data'].data[i:i+1,0].transpose(1,0,2).reshape(28,28)
	#     im2   = solver.net.blobs['data_p'].data[i:i+1,0].transpose(1,0,2).reshape(28,28)
	#     Pair  = np.concatenate((im1, im2), axis=1)
	#     label = solver.net.blobs['sim'].data[i:i+1]
	#
	#     if (label > 0):
	#         name = "data/image_pairs/Class1/Pair_" + str(i) + "--Label_" + str(label[0]) + ".png"
	#     else:
	#         name = "data/image_pairs/Class0/Pair_" + str(i) + "--Label_" + str(label[0]) + ".png"
	#
	#     imsave(name, Pair)


def saveTriplets(NET, TEST, train_batch, folder, num_batch):
	for i in range(num_batch):
		NET.forward()  # train net
		TEST.forward()  # test net (there can be more than one)

	im1_path = open('./data/triplets/train2.txt')  # reference image
	im1_lines = im1_path.readlines()
	im2_path = open('./data/triplets/train0.txt')  # positive instance
	im2_lines = im2_path.readlines()
	im3_path = open('./data/triplets/train1.txt')  # negative instance
	im3_lines = im3_path.readlines()

	for i in range(train_batch):
		im1 = NET.blobs['data_n'].data[i]
		im1 = ndarray.transpose(im1, (1, 2, 0))
		im2 = NET.blobs['data'].data[i]
		im2 = ndarray.transpose(im2, (1, 2, 0))
		im3 = NET.blobs['data_p'].data[i]
		im3 = ndarray.transpose(im3, (1, 2, 0))
		Triplet = np.concatenate((im1, im2, im3), axis=1)
		Triplet = misc.toimage(Triplet)
		# Triplet = Triplet.rotate(-180)
		Triplet = misc.fromimage(Triplet)

		line = im1_lines[((num_batch - 1) * train_batch) + i]
		path, _ = line.split(' ')
		print(path)
		im1_txt = imread(path)
		line = im2_lines[((num_batch - 1) * train_batch) + i]
		path, _ = line.split(' ')
		im2_txt = imread(path)
		line = im3_lines[((num_batch - 1) * train_batch) + i]
		path, _ = line.split(' ')
		im3_txt = imread(path)
		Triplet_txt = np.concatenate((im1_txt, im2_txt, im3_txt), axis=1)

		images = np.concatenate((Triplet, Triplet_txt), axis=0)

		imsave(folder + 'Triplet_' + str(i) + '.jpg', images)

	im1_path.close()
	im2_path.close()
	im3_path.close()


def computeAcc(Labels, Predictions, test_batch, test_interval, test_it, it, test_acc, correct):
	correct = correct + sum(Labels == Predictions)

	test_acc[it // test_interval] = float(correct) / float(test_batch * test_it)

	return test_acc


def transformImage(image, Train=1):
	if Train:
		# (3, 227, 227) -> (277, 277, 3)
		image = np.transpose(image, (1, 2, 0))
		# BGR -> RGB
		image = image[:, :, (2, 1, 0)]
		image = image.astype(np.uint8)
	else:
		# BGR -> RGB
		image = image[:, :, (2, 1, 0)]
		image = image.astype(np.uint8)

	return image


def createTXTfiles(MEME_images, num_train, folder_files, flag, net):
	train1 = open(folder_files + flag + '1.txt', 'w')
	train2 = open(folder_files + flag + '2.txt', 'w')

	if net == 'siamese':

		for i in range(0, num_train, 1):
			folders = os.listdir(MEME_images)
			# Positive pair
			folder = choice(folders)
			path = MEME_images + folder + '/'
			memes = os.listdir(path)
			pos_pair1 = choice(memes)
			train1.write(path + pos_pair1 + ' ' + str(1) + '\n')
			pos_pair2 = choice(memes)
			train2.write(path + pos_pair2 + ' ' + str(1) + '\n')

			# Negative pair
			folder = choice(folders)
			path = MEME_images + folder + '/'
			memes = os.listdir(path)
			neg_pair1 = choice(memes)
			train1.write(path + neg_pair1 + ' ' + str(0) + '\n')
			folders.remove(folder)
			folder = choice(folders)
			path = MEME_images + folder + '/'
			memes = os.listdir(path)
			neg_pair2 = choice(memes)
			train2.write(path + neg_pair2 + ' ' + str(0) + '\n')

	elif net == 'triplet':

		train0 = open(folder_files + flag + '0.txt', 'w')

		for i in range(0, num_train, 1):
			aux_folders = os.listdir(MEME_images)
			# Reference image
			folder = choice(aux_folders)
			path = MEME_images + folder + '/'
			memes = os.listdir(path)
			ref_img = choice(memes)
			train0.write(path + ref_img + ' ' + str(2) + '\n')

			# Positive instance
			memes.remove(ref_img)
			pos_ins = choice(memes)
			train1.write(path + pos_ins + ' ' + str(1) + '\n')

			# Negative instance
			aux_folders.remove(folder)
			folder = choice(aux_folders)
			path = MEME_images + folder + '/'
			memes = os.listdir(path)
			neg_ins = choice(memes)
			train2.write(path + neg_ins + ' ' + str(0) + '\n')

		train0.close()

	train1.close()
	train2.close()


def createFeatureMatrix(feat_matrix, NET, train_batch, it):
	# filters = NET.params['conv1'][0].data
	feat = NET.blobs['feat_norm'].data
	feat_n = NET.blobs['feat_false_norm'].data
	feat_p = NET.blobs['feat_true_norm'].data
	# saveFilters(filters, 'conv1', test_it, feat, name)

	# Labels
	labels = NET.blobs['sim'].data
	labels_n = NET.blobs['sim_n'].data
	labels_p = NET.blobs['sim_p'].data

	# images
	img = NET.blobs['data'].data
	img_n = NET.blobs['data_n'].data
	img_p = NET.blobs['data_p'].data

	images = []
	label_matrix = []
	dim = feat[0].shape[0]
	for i in range(train_batch):
		feat_matrix.append(feat[i].reshape(dim))
		label_matrix.append(labels[i])
		images.append(img[i])

		feat_matrix.append(feat_n[i].reshape(dim))
		label_matrix.append(labels_n[i])
		images.append(img_n[i])

		feat_matrix.append(feat_p[i].reshape(dim))
		label_matrix.append(labels_p[i])
		images.append(img_p[i])

	prediction_example = feat_p[i].reshape(dim)

	return feat_matrix, label_matrix, prediction_example, dim, images


def load_mnist(dataset="training", digits=np.arange(10), path="."):
	"""
	Loads MNIST files into 3D numpy arrays

	Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
	"""

	if dataset is "training":
		fname_img = os.path.join(path, 'train-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
	elif dataset is "testing":
		fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
	else:
		raise ValueError, "dataset must be 'testing' or 'training'"

	flbl = open(fname_lbl, 'rb')
	magic_nr, size = struct.unpack(">II", flbl.read(8))
	lbl = pyarray("b", flbl.read())
	flbl.close()

	fimg = open(fname_img, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = pyarray("B", fimg.read())
	fimg.close()

	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)

	images = zeros((N, rows, cols), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i in range(len(ind)):
		images[i] = np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
		labels[i] = lbl[ind[i]]

		# im = np.array([np.tile(images[i], (1, 1)) for j in xrange(3)])
		# aux = misc.toimage(im)
		# aux = misc.toimage(images[i])
		#
		# name = "/home/olaia.artieda/TFM/DATASETS/MNIST/" + dataset + "/" + str(labels[i, 0]) + "/" + str(i) + ".png"
		# aux.save(name)

	return images, labels


def load_images(path):
	folders = os.listdir(path)
	txt_file = open('./data/triplets/curves.txt', 'w')

	num_images = 0
	images = []
	labels = []

	for folder in folders:
		class_path = path + folder + '/'
		class_images = os.listdir(class_path)

		for image in class_images:
			txt_file.write(class_path + image + ' ' + str(folder) + '\n')

			image = imread(class_path + image)
			images.append(image)
			labels.append(folder)

			num_images += 1

	txt_file.close()

	return txt_file, images, labels


def unpickle(file):
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict


def load_cifar10(dataset_path, aux):
	images = []
	norm_images = []
	labels = []

	image_size = 32
	mean_image = np.zeros((image_size, image_size, 3))
	mean = []

	# nearest neighbors
	testing_labels = 0
	testing_images = []

	if aux == 'training':
		for i in range(1, 6, 1):
			dict = unpickle(dataset_path + 'data_batch_' + str(i))

			num_images = dict['data'].shape[0]
			for ii in range(num_images):
				img = dict['data'][ii].reshape((3, image_size, image_size))
				label = dict['labels'][ii]
				rgb_img = ndarray.transpose(img, (2, 1, 0))
				# image = misc.toimage(rgb_img)
				# image = image.rotate(-90)
				# image.save(dataset_path + aux + '/' + str(label) + '/' + str(ii) + '.png')
				images.append(rgb_img)
				labels.append(label)

				# Create mean image
				mean_image[:, :, 0] += rgb_img[:, :, 0]
				mean_image[:, :, 1] += rgb_img[:, :, 1]
				mean_image[:, :, 2] += rgb_img[:, :, 2]

			mean.append(mean_image / ii*i)

	elif aux == 'testing':
		dict = unpickle(dataset_path + 'test_batch')
		num_images = dict['data'].shape[0]
		for ii in range(num_images):
			img = dict['data'][ii].reshape((3, image_size, image_size))
			label = dict['labels'][ii]
			rgb_img = ndarray.transpose(img, (2, 1, 0))
			# image = misc.toimage(rgb_img)
			# image = image.rotate(-90)
			# image.save(dataset_path + aux + '/' + str(label) + '/' + str(ii) + '.png')
			images.append(rgb_img)
			labels.append(label)

			# Create mean image
			mean_image[:, :, 0] += rgb_img[:, :, 0]
			mean_image[:, :, 1] += rgb_img[:, :, 1]
			mean_image[:, :, 2] += rgb_img[:, :, 2]

		mean.append(mean_image / ii)

	return images, labels, mean


def save_images_txt(path, file_path):
	folders = os.listdir(path)
	txt_file = open(file_path, 'w')

	for folder in folders:
		class_path = path + folder + '/'
		class_images = os.listdir(class_path)

		for image in class_images:
			txt_file.write(class_path + image + ' ' + str(folder) + '\n')

	txt_file.close()

	return txt_file


def modifyTrainTestFile(const, alpha_param, feats):
	if const == 'mnist':
		file_path = './data/' + const + '_train_test.prototxt'
		f = open(file_path, 'r')
		contents = f.readlines()
		f.close()

		# Alpha margin
		index = 605
		value = "    param_str: " + '"' + "'" + "num" + "': " + str(alpha_param) + '"' + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		# Features
		index = 241
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 398
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 555
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)

		f = open(file_path, 'w')
		contents = "".join(contents)
		f.write(contents)
		f.close()

		#############################################################################3

		file_path = './data/' + const + '_net_curves.prototxt'
		f = open(file_path, 'r')
		contents = f.readlines()
		f.close()

		# Alpha margin
		index = 605
		value = "    param_str: " + '"' + "'" + "num" + "': " + str(alpha_param) + '"' + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		# Features
		index = 241
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 398
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 555
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)

		f = open(file_path, 'w')
		contents = "".join(contents)
		f.write(contents)
		f.close()

	elif const == 'cifar10':
		file_path = './data/' + const + '_train_test.prototxt'
		f = open(file_path, 'r')
		contents = f.readlines()
		f.close()

		# Alpha margin
		index = 605
		value = "    param_str: " + '"' + "'" + "num" + "': " + str(alpha_param) + '"' + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		# Features
		index = 241
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 398
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 555
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)

		f = open(file_path, 'w')
		contents = "".join(contents)
		f.write(contents)
		f.close()

		#############################################################################3

		file_path = './data/' + const + '_net_curves.prototxt'
		f = open(file_path, 'r')
		contents = f.readlines()
		f.close()

		# Alpha margin
		index = 605
		value = "    param_str: " + '"' + "'" + "num" + "': " + str(alpha_param) + '"' + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		# Features
		index = 241
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 398
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)
		index = 555
		value = "    num_output: " + str(feats) + "\n"
		contents.insert(index, value)
		contents.pop(index + 1)

		f = open(file_path, 'w')
		contents = "".join(contents)
		f.write(contents)
		f.close()
