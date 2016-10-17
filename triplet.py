import os
import caffe
from pylab import *
from sklearn.neighbors import NearestNeighbors
from functions import createTXTfiles, pretrainedEmbedding, modifyTrainTestFile, save_images_txt, createFeatureMatrix
from graphics import load_mnist, load_cifar10, plotNeighbors, plotMeanCurves, plotHistogram, plotLoss, \
	visualizeEmbedding


def main():
	caffe_root = '/home/olaia.artieda/caffe-master/'
	sys.path.insert(0, caffe_root + 'python')
	sys.path.insert(0, '/home/olaia.artieda/EURECAT_project/Triplet/Triplet_network')

	caffe.set_device(0)
	caffe.set_mode_gpu()

	####################################################################################################
	# VARIABLES
	pretrained = False
	dataset = 'mnist'
	network = 'caffenet'
	train_niter = 5000
	test_interval = 1000
	test_nimages = 10000
	train_images = 60000

	RESULTS_DIR = '../../results/triplet_' + network + '_' + dataset + '/'
	SOLVER_DIR = './data/' + network + '_' + dataset + '_solver.prototxt'

	cifar10_dataset_path = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/'
	cifar10_dataset_path_train = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/training_227/'
	cifar10_dataset_path_val = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/testing_227/'

	mnist_dataset_path = '/home/olaia.artieda/TFM/DATASETS/MNIST/'
	mnist_dataset_path_train = '/home/olaia.artieda/TFM/DATASETS/original_MNIST/training_227/'
	mnist_dataset_path_val = '/home/olaia.artieda/TFM/DATASETS/original_MNIST/testing_227/'

	####################################################################################################
	# CREATE TXT FILES
	if dataset == 'cifar10':
		# # TRAIN AND VALIDATION FILES
		# file_path = './data/triplets/'
		# file_name = network + '_' + dataset + '_' + 'train_'
		# num_images = train_images / 3
		# createTXTfiles(cifar10_dataset_path_train, num_images, file_path, file_name)
		# file_name = network + '_' + dataset + '_' + 'val_'
		# num_images = test_nimages / 3
		# createTXTfiles(cifar10_dataset_path_val, num_images, file_path, file_name)

		# TEST FILE
		# train_images, train_labels, train_mean = load_cifar10(cifar10_dataset_path, 'training')
		# test_images, test_labels, test_mean = load_cifar10(cifar10_dataset_path, 'testing')
		file_path = './data/triplets/cifar10_val.txt'
		# _ = save_images_txt(cifar10_dataset_path_val, file_path)

	elif dataset == 'mnist':
		# # TRAIN AND VALIDATION FILES
		# file_path = './data/triplets/'
		# file_name = network + '_' + dataset + '_' + 'train_'
		# num_images = train_images / 3
		# createTXTfiles(mnist_dataset_path_train, num_images, file_path, file_name)
		# file_name = network + '_' + dataset + '_' + 'val_'
		# num_images = test_nimages / 3
		# createTXTfiles(mnist_dataset_path_val, num_images, file_path, file_name)

		# TEST FILE
		# train_images, train_labels = load_mnist('training', digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		#                                         path=mnist_dataset_path)
		# test_images, test_labels = load_mnist('testing', digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
		#                                       path=mnist_dataset_path)
		file_path = './data/triplets/mnist_val.txt'
		# _ = save_images_txt(mnist_dataset_path_val, file_path)

	####################################################################################################

	if pretrained:
		# PRETRAINED MODEL
		model_def = './data/caffenet_cifar10_train_test.prototxt'
		model_weights = '/home/olaia.artieda/Desktop/TrainedModels/Iteration_200000.caffemodel'

		test_batch = 8
		n_neighbours = 10
		image_path = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/testing/13.png'
		pretrainedEmbedding(model_def, model_weights, RESULTS_DIR, image_path, test_batch, n_neighbours, dataset,
		                    cifar10_dataset_path_val)

	elif not pretrained:
		solver = caffe.SGDSolver(SOLVER_DIR)
		TRAIN = solver.net
		TEST = solver.test_nets[0]

		train_batch = TRAIN.blobs['data'].num
		test_batch = TEST.blobs['data'].num
		num_batches = test_nimages / test_batch

		Loss_train = zeros(train_niter)
		Loss_val = zeros(train_niter)

		# checkInputData(solver)

		for feat_pow in range(3, 4, 2):  # mnist: 128, cifar10: 8
			feat_quan = 2 ** feat_pow
			alpha_AUC = []

			for alpha in range(2, 4, 1):  # mnist: 0.2, cifar10: ???
				mean = []
				alpha_param = alpha / 10.
				modifyTrainTestFile(dataset, alpha_param, feat_quan)

				for experiment in range(1, 2, 1):
					# Create folders
					directory = RESULTS_DIR + 'feats_' + str(feat_quan) + '/' + 'alpha_' + str(alpha_param) + '/' + str(
						experiment) + '/'
					if not os.path.exists(directory):
						os.makedirs(directory)

					for it in range(train_niter):
						solver.step(1)
						Loss_train[it] = solver.net.blobs['loss'].data

						if it % test_interval == 0:
							print 'Iteration', it, 'testing...'

							loss = 0
							All_Ds_p_layer = np.zeros(test_nimages / 3)
							All_Ds_n_layer = np.zeros(test_nimages / 3)
							feat_matrix = []
							images = []
							label_matrix = []

							[feat_matrix, _, _, dim, images] = createFeatureMatrix(feat_matrix, TRAIN, train_batch, it)

							# KNN
							n_neighbors = 10
							neigh = NearestNeighbors(n_neighbors)
							neigh.fit(feat_matrix)

							for test_it in range(0, num_batches, 1):
								TEST.forward()  # one batch

								loss += TEST.blobs['loss'].data

								D_p_layer = TEST.blobs['distance_p'].data
								D_n_layer = TEST.blobs['distance_n'].data
								All_Ds_p_layer[test_it * test_batch: (test_it + 1) * test_batch] = D_p_layer
								All_Ds_n_layer[test_it * test_batch: (test_it + 1) * test_batch] = D_n_layer

							Loss_val[it] = loss / test_it

							# Histogram of distances
							name = directory + 'DistanceHistLayer_it_' + str(it) + '.jpg'
							plotHistogram(All_Ds_p_layer, All_Ds_n_layer, name)

							# Triplet loss
							name = directory + 'Loss_it_' + str(it) + '.jpg'
							plotLoss(Loss_train, Loss_val, train_niter, name)

				# # KNN
				# for i in range(0, test_batch, 1):
				# 	# images from the last testing batch
				# 	feats = TEST.blobs['feat'].data[i]
				# 	im = TEST.blobs['data'].data[i]
				# 	# Ask who is the closest images to feats
				# 	[dist, ind] = neigh.kneighbors(feats.reshape(dim))
				# 	folder = directory + 'neighbors/' + str(it) + '/'
				# 	if not os.path.exists(folder):
				# 		os.makedirs(folder)
				# 	plotNeighbors(dataset, im, ind[0], images, n_neighbors, folder, i, it)
				#
				# #########################################################################################
				# # Plot precision and recall curves
				# MODEL = './data/snapshots/' + dataset + '_triplet_iter_' + str(train_niter) + '.caffemodel'
				# NETWORK = './data/' + dataset + '_net_curves.prototxt'
				# path_features = directory + 'feature_matrix.png'
				# path_curves = directory + 'meanPR_AUC_curves.png'
				# if dataset == 'cifar10':
				# 	val_set = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/testing/'
				# elif dataset == 'mnist':
				# 	val_set = '/home/olaia.artieda/TFM/DATASETS/MNIST/testing/'
				#
				# class_AUC = plotMeanCurves(MODEL, NETWORK, path_features, path_curves, val_set)
				# mean.append(sum(class_AUC) / float(len(class_AUC)))
				#
				# #########################################################################################
				# # Plot final embedding
				# net = caffe.Net(NETWORK, MODEL, caffe.TEST)
				# fig_path = directory + 'final_embedding_'
				# visualizeEmbedding(net, 'feat', fig_path, num_feats=2)
				# visualizeEmbedding(net, 'feat_norm', fig_path, num_feats=2)

				# Plot alpha curves
				alpha_AUC.append(sum(mean) / float(len(mean)))

			# Save values
			file_path = directory + dataset + '_meanAUC.txt'
			f = open(file_path, 'w')
			f.writelines("%s\n" % item for item in alpha_AUC)
			f.close()


if __name__ == "__main__":
	main()
