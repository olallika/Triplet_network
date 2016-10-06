import os
import caffe
from pylab import *
from sklearn.neighbors import NearestNeighbors
from graphics import plotHistogram, plotLoss, checkInputData, visualizeEmbedding, plotNeighbors, plotMeanCurves
from functions import createTXTfiles, load_mnist, load_cifar10, save_images_txt, createFeatureMatrix, modifyTrainTestFile

caffe_root = '/home/olaia.artieda/caffe-master/'
sys.path.insert(0, '/home/olaia.artieda/Desktop/caffe-master')
sys.path.insert(0, '/home/olaia.artieda/TFM/TripletMNIST/')
sys.path.insert(0, caffe_root + 'python')

caffe.set_device(0)
caffe.set_mode_gpu()


# Draw net
# ./python/draw_net.py /home/olaia.artieda/TFM/triplet_network/data/meme_triplet_train_test.prototxt /home/olaia.artieda/TFM/triplet_network/triplet.png

def main():
	# Modify the feature quantity in the last fully connected layer
	for feat_pow in range(3, 4, 2): # mnist: 128, cifar10: 8
		feat_quan = 2 ** feat_pow

		# Variables
		alpha_AUC = []
		const = 'cifar10'

		for alpha in range(2, 4, 1): # mnist: 0.2, cifar10: ???
			# Variables
			mean = []

			#########################################################################################
			# Modify feature quantity and alpha margin
			alpha_param = alpha / 10.
			modifyTrainTestFile(const, alpha_param, feat_quan)

			for experiment in range(1, 2, 1):
				# Create folders
				directory = './data/results/feats_' + str(feat_quan) + '/' + 'alpha_' + str(alpha_param) + '/' + str(experiment) + '/'
				if not os.path.exists(directory):
					os.makedirs(directory)


				#########################################################################################
				if const == 'cifar10':
					# Save all Cifar10 images in folders
					dataset_path = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/'
					train_images, train_labels, train_mean = load_cifar10(dataset_path, 'training')
					test_images, test_labels, test_mean = load_cifar10(dataset_path, 'testing')
					# Cifar10
					val_path = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/testing/'
					file_path = './data/triplets/cifar10_val.txt'
					_ = save_images_txt(val_path, file_path)
				elif const == 'mnist':
					# Save all mnist images in folders
					train_images, train_labels = load_mnist('training', digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], path="/home/olaia.artieda/TFM/DATASETS/MNIST/")
					test_images, test_labels = load_mnist('testing', digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], path="/home/olaia.artieda/TFM/DATASETS/MNIST/")
					# Create txt files with validation images and labels
					# MNIST
					val_path = '/home/olaia.artieda/TFM/DATASETS/MNIST/testing/'
					file_path = './data/triplets/mnist_val.txt'
					# _ = save_images_txt(val_path, file_path)


				#########################################################################################
				if const == 'cifar10':
					train_images = 40000 * 3  # 30000*3 triplets
					test_images = 3337 * 3  # 5000*3 triplets
					triplets_train = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/training/'
					triplets_val = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/testing/'
					# createTXTfiles(triplets_train, train_images / 3, './data/triplets/', 'trainCifar10', 'triplet')
					# createTXTfiles(triplets_val, test_images / 3, './data/triplets/', 'valCifar10', 'triplet')
				elif const == 'mnist':
					train_images = 36244 * 3  # 30000*3 triplets
					test_images = 3337 * 3  # 5000*3 triplets
					# triplets_train = '/home/olaia.artieda/TFM/DATASETS/MNIST/training/'
					# triplets_val = '/home/olaia.artieda/TFM/DATASETS/MNIST/testing/'
					# createTXTfiles(triplets_train, train_images / 3, './data/triplets/', 'train', 'triplet')
					# createTXTfiles(triplets_val, test_images / 3, './data/triplets/', 'val', 'triplet')

				#########################################################################################
				solver = caffe.SGDSolver('data/' + const + '_triplet_solver.prototxt')
				NET = solver.net
				TEST = solver.test_nets[0]

				niter = 10000
				test_interval = 1000
				test_batch = solver.test_nets[0].blobs['data'].num
				train_batch = solver.net.blobs['data'].num
				num_batches = int((test_images / 3) / test_batch)
				# niter = int((train_images / 3) / train_batch)
				Loss_train = zeros(niter)
				Loss_val = zeros(niter)

				#########################################################################################
				# for layer_name, blob in NET.blobs.iteritems():
				# 	print layer_name + '\t' + str(blob.data.shape)
				#
				# for layer_name, param in NET.params.iteritems():
				# 	print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


				#########################################################################################
				# Before taking off, lt's check that everything is loaded as we expect.
				# checkInputData(solver)


				#########################################################################################
				for it in range(niter):
					solver.step(1)

					Loss_train[it] = solver.net.blobs['loss'].data

					if it % test_interval == 0:
						print 'Iteration', it, 'testing...'

						loss = 0

						All_Ds_p_layer = np.zeros(test_images / 3)
						All_Ds_n_layer = np.zeros(test_images / 3)

						feat_matrix = []
						images = []
						label_matrix = []

						[feat_matrix, _, _, dim, images] = createFeatureMatrix(feat_matrix, NET, train_batch, it)

						# KNN
						n_neighbors = 10
						neigh = NearestNeighbors(n_neighbors)
						neigh.fit(feat_matrix)

						for test_it in range(0, num_batches, 1):
							TEST.forward()  # one batch

							loss += solver.test_nets[0].blobs['loss'].data

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
						plotLoss(Loss_train, Loss_val, niter, name)

			# KNN
			for i in range(0, test_batch, 1):
				# images from the last testing batch
				feats = TEST.blobs['feat'].data[i]
				im = TEST.blobs['data'].data[i]
				# Ask who is the closest images to feats
				[dist, ind] = neigh.kneighbors(feats.reshape(dim))
				folder = directory + 'neighbors/' + str(it) + '/'
				if not os.path.exists(folder):
					os.makedirs(folder)
				plotNeighbors(const, im, ind[0], images, n_neighbors, folder, i, it)

			#########################################################################################
			# Plot precision and recall curves
			MODEL = './data/snapshots/' + const + '_triplet_iter_' + str(niter) + '.caffemodel'
			NETWORK = './data/' + const + '_net_curves.prototxt'
			path_features = directory + 'feature_matrix.png'
			path_curves = directory + 'meanPR_AUC_curves.png'
			if const == 'cifar10':
				val_set = '/home/olaia.artieda/TFM/DATASETS/cifar-10-batches-py/testing/'
			elif const == 'mnist':
				val_set = '/home/olaia.artieda/TFM/DATASETS/MNIST/testing/'

			class_AUC = plotMeanCurves(MODEL, NETWORK, path_features, path_curves, val_set)
			mean.append(sum(class_AUC) / float(len(class_AUC)))

			#########################################################################################
			# Plot final embedding
			net = caffe.Net(NETWORK, MODEL, caffe.TEST)
			fig_path = directory + 'final_embedding_'
			visualizeEmbedding(net, 'feat', fig_path, num_feats=2)
			visualizeEmbedding(net, 'feat_norm', fig_path, num_feats=2)

			# Plot alpha curves
			alpha_AUC.append(sum(mean) / float(len(mean)))

		# Save values
		file_path = directory + const + '_meanAUC.txt'
		f = open(file_path, 'w')
		f.writelines("%s\n" % item for item in alpha_AUC)
		f.close()


if __name__ == "__main__":
	main()
