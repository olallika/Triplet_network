import struct
from array import array as pyarray
import cPickle
import caffe
from random import choice
from pylab import *
import numpy as np
import scipy.misc as misc
import os
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from graphics import plotMeanCurves, plotNeighbors


def pretrainedEmbedding(model_def, model_weights, RESULTS_DIR, image_path, test_batch, n_neighbours, dataset,
                        dataset_path_val):
	network = caffe.Net(model_def, model_weights, caffe.TEST)

	# K-NEAREST-NEIGHBOURS
	feat_matrix = []
	images = []
	label_matrix = []
	it = 0
	[feat_matrix, _, _, dim, images] = createFeatureMatrix(feat_matrix, images, label_matrix, network, test_batch)
	neigh = NearestNeighbors(n_neighbours)
	neigh.fit(feat_matrix)

	# load image and prepare as a single input batch for Caffe
	im = np.array(Image.open(image_path))
	im_input = im[np.newaxis, np.newaxis, :, :]
	network.blobs['data'].reshape(*im_input.shape)
	network.blobs['data'].data[...] = im_input

	network.forward()

	for i in range(0, test_batch, 1):
		# images from the last testing batch
		feats = network.blobs['feat'].data[i]
		im = network.blobs['data'].data[i]
		# Ask who is the closest images to feats
		[dist, ind] = neigh.kneighbors(feats.reshape(dim))
		folder = RESULTS_DIR + 'siamese_neighbors/' + str(it) + '/'
		if not os.path.exists(folder):
			os.makedirs(folder)
		plotNeighbors(dataset, im, ind[0], images, n_neighbours, folder, i, it)

	# Plot precision and recall curves
	path_curves = RESULTS_DIR + dataset + 'meanPR_AUC_curves.png'
	class_AUC = plotMeanCurves(model_weights, model_def, path_curves, dataset_path_val)

	return class_AUC


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


def createTXTfiles(images_path, num_images, file_path, file_name):
	train0 = open(file_path + file_name + '0.txt', 'w')
	train1 = open(file_path + file_name + '1.txt', 'w')
	train2 = open(file_path + file_name + '2.txt', 'w')

	for i in range(0, num_images, 1):
		aux_folders = os.listdir(images_path)
		# Reference image
		folder = choice(aux_folders)
		path = images_path + folder + '/'
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
		path = images_path + folder + '/'
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


def changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path):
	f = open(file_path, 'r')
	contents = f.readlines()
	f.close()

	# Alpha margin
	value = "    margin: " + str(alpha_param) + "\n"
	contents.insert(index1, value)
	contents.pop(index1 + 1)

	# Features
	value = "    num_output: " + str(feats) + "\n"
	contents.insert(index2, value)
	contents.pop(index2 + 1)
	value = "    num_output: " + str(feats) + "\n"
	contents.insert(index3, value)
	contents.pop(index3 + 1)
	value = "    num_output: " + str(feats) + "\n"
	contents.insert(index4, value)
	contents.pop(index4 + 1)

	f = open(file_path, 'w')
	contents = "".join(contents)
	f.write(contents)
	f.close()


def modifyTrainTestFile(dataset, network, alpha_param, feats):
	if dataset == 'mnist' and network == 'lenet':
		index1 = 605
		index2 = 241
		index3 = 398
		index4 = 555
		file_path = './data/' + network + '_' + dataset + '_train_test.prototxt'
		changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path)
		file_path = './data/' + network + '_' + dataset + '_net_curves.prototxt'
		changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path)

	elif dataset == 'cifar10' and network == 'lenet':
		index1 = 605
		index2 = 241
		index3 = 398
		index4 = 555
		file_path = './data/' + network + '_' + dataset + '_train_test.prototxt'
		changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path)
		file_path = './data/' + network + '_' + dataset + '_net_curves.prototxt'
		changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path)

	elif dataset == 'cifar10' and network == 'caffenet':
		index1 = 605
		index2 = 241
		index3 = 398
		index4 = 555
		file_path = './data/' + network + '_' + dataset + '_train_test.prototxt'
		changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path)
		file_path = './data/' + network + '_' + dataset + '_net_curves.prototxt'
		changeMarginFeats(alpha_param, feats, index1, index2, index3, index4, file_path)
