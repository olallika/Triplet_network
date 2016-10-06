import caffe
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
from evaluation import PR_curve
from functions import transformImage, load_mnist
from scipy.spatial import distance
from functions import load_images
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def saveAcc(total_iter, train_loss, test_interval, test_acc, col, it, const, actual_epoch):
	fig, ax1 = plt.subplots()
	ax1.plot(arange(total_iter), train_loss, 'k')
	ax1.set_xlabel('Iteration')
	ax1.set_ylabel('Contrastive Loss', color='k')
	for tl in ax1.get_yticklabels():
		tl.set_color('k')

	ax2 = ax1.twinx()
	ax2.plot(test_interval * arange(len(test_acc)), test_acc, col)
	ax2.set_ylabel('Accuracy', color=col)
	for tl in ax2.get_yticklabels():
		tl.set_color(col)
	plt.title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
	plt.savefig('./data/results/TrainingAndTesting_' + const + '_it_' + str(it) + '_epoch_' + str(
		actual_epoch) + '.png')
	plt.close()


def savePR(recall, precision, col, const, it, actual_epoch):
	plt.figure()
	plt.plot(recall, precision, col)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve')
	plt.savefig('./data/results/PrecisionRecall_' + const + '_' + str(it) + '_epoch_' + str(actual_epoch) + '.png')
	plt.close()


def covarianceMatrix(features, normalized_features, name, it):
	cov_feats = np.cov(features)
	cov_norm_feats = np.cov(normalized_features)

	plt.figure('Covariance matrices')
	plt.subplot(121)
	plt.title('fc8 features')
	plt.imshow(cov_feats)
	plt.colorbar()
	plt.axis('off')
	plt.subplot(122)
	plt.title('fc8 normalized features')
	plt.imshow(cov_norm_feats)
	plt.colorbar()
	plt.axis('off')
	plt.draw()

	plt.savefig(name + '_it' + str(it) + '.jpg')
	plt.close()


def transformImage(image, Train=1):
	if Train:
		# (3, 227, 227) -> (277, 277, 3)
		image = np.transpose(image, (1, 2, 0))
		# BGR -> RGB
		image = image[:, :, (2, 1, 0)]
	else:
		image = np.transpose(image, (1, 2, 0))
		image = image[:, :, (2, 1, 0)]

	return image


def plotNeighbors(const, test_im, neighbors, test_images, n_neighbors, folder, batch, it):
	# plot_rows = (n_neighbors / 5) + 1
	plot_rows = 1
	plt.figure('Nearest neighbors')
	plt.subplot(plot_rows, 11, 1)
	# plt.title('Test image')
	image = transformImage(test_im, Train=1)
	if const == 'cifar10':
		plt.imshow(image+abs(image.min()), )
	elif const == 'mnist':
		plt.imshow(image, )
	plt.axis('off')

	# Retrieve the 128 images

	for i in range(0, n_neighbors):
		plt.subplot(plot_rows, 11, 2 + i)
		img = test_images[neighbors[i]]
		image = transformImage(img, Train=0)
		if const == 'cifar10':
			plt.imshow(image + abs(image.min()), )
		elif const == 'mnist':
			plt.imshow(image, )
		plt.axis('off')

	plt.draw()
	# show(block=True)
	savefig(folder + 'neighbors_ it' + str(it) + '_batchImg_' + str(batch) + '.jpg')
	plt.close()


def checkInputData(solver):
	# Before taking off, let's check that everything is loaded as we expect.
	# We'll run a forward pass on the train and test nets and check that they contain our data.
	solver.net.forward()  # train net
	solver.test_nets[0].forward()  # test net (there can be more than one)

	# We take the train batch size. Get all pairs of images from the first batch and save them together within a folder 'data/image_pairs/'
	# In this folder we distiguish among pairs of the same class and pairs of different classes
	# Note that class=1 means being from the same class (while in LeCun's original paper is the other way around!)
	train_batch_size = solver.net.blobs['data'].data.shape[0]
	for i in arange(train_batch_size):
		# Get the first pair of images and its label
		im1 = solver.net.blobs['data'].data[i]
		im1 = ndarray.transpose(im1, (2, 1, 0))
		im2 = solver.net.blobs['data_p'].data[i]
		im2 = ndarray.transpose(im2, (2, 1, 0))
		im3 = solver.net.blobs['data_n'].data[i]
		im3 = ndarray.transpose(im3, (2, 1, 0))

		Triplet = np.concatenate((im1, im2, im3), axis=1)
		Triplet = misc.toimage(Triplet)
		# Triplet = Triplet.rotate(-90)
		Triplet = misc.fromimage(Triplet)

		name = "data/image_triplets/Triplet_" + str(i) + ".png"
		imsave(name, Triplet)
		plt.close()


def savePairs(NET, TEST, train_batch, folder, num_batch):
	for i in range(num_batch):
		NET.forward()  # train net
		TEST.forward()  # test net (there can be more than one)

	im1_path = open('./data/pairs/train1.txt')  # reference image
	im1_lines = im1_path.readlines()
	im2_path = open('./data/pairs/train2.txt')  # positive instance
	im2_lines = im2_path.readlines()

	for i in range(train_batch):
		im1 = NET.blobs['data_p'].data[i]
		im2 = NET.blobs['data'].data[i]
		Triplet = np.concatenate((im1, im2), axis=1)
		Triplet = transformImage(Triplet, Train=0)

		line = im1_lines[((num_batch - 1) * train_batch) + i]
		path, _ = line.split(' ')
		im1_txt = imread(path)
		line = im2_lines[((num_batch - 1) * train_batch) + i]
		path, _ = line.split(' ')
		im2_txt = imread(path)
		Triplet_txt = np.concatenate((im1_txt, im2_txt), axis=1)

		images = np.concatenate((Triplet, Triplet_txt), axis=0)

		imsave(folder + 'Pair_' + str(i) + '.jpg', images)

	im1_path.close()
	im2_path.close()


def plotEmbedding(X_ref, X_p, X_n, name):
	n_clusters = 3
	plt.figure()
	colors = ['r', 'b', 'g']

	# Features
	X = np.vstack((X_ref, X_p))
	X = np.vstack((X, X_n))
	# Labels
	lab_ref = ones((X_ref.shape[0], 1)) * 2
	lab_p = ones((X_p.shape[0], 1))
	lab_n = zeros((X_n.shape[0], 1))
	k_means_labels = np.vstack((lab_ref, lab_p))
	k_means_labels = np.vstack((k_means_labels, lab_n))
	k_means_labels = int32(k_means_labels)
	k_means_labels = k_means_labels.flatten()

	for k, col in zip(range(n_clusters), colors):
		my_members = k_means_labels == k
		plt.scatter(X[my_members, 0], X[my_members, 1], s=80, facecolors='none', edgecolors=col)

	plt.title('2D Feature Space')
	plt.savefig(name)
	plt.close()


def plotLoss(Loss_train, Loss_val, it, name):
	plt.figure()
	# axes = plt.gca()
	# axes.set_ylim([0, 30])
	x = arange(it)
	plt.plot(x, Loss_train, 'b')
	plt.plot(x, Loss_val, 'r')
	plt.xlabel('Train iterations')
	plt.ylabel('Loss')
	plt.title('Triplet Loss')
	plt.legend(['Train loss', 'Val loss'])
	plt.savefig(name)
	plt.close()


def visualizeEmbedding(net, features, fig_path, num_feats):
	val_set = '/home/olaia.artieda/TFM/DATASETS/MNIST/testing/'
	_, test_images, test_labels = load_images(val_set)

	num_images = len(test_images)
	dim_images = test_images[0].shape[1]
	# num_feats = net.blobs['feat_norm'].data.shape[1]
	batch_size = net.blobs['feat_norm'].data.shape[0]
	num_batches = num_images / batch_size

	if num_feats == 2:
		f = plt.figure(figsize=(16, 9))
	elif num_feats == 3:
		f = plt.figure()
		ax = Axes3D(f, elev=-150, azim=110)

	for i in range(num_batches):
		net.forward()

		# Apply PCA
		feat = net.blobs[features].data
		pca = PCA(num_feats)
		new_feats = pca.fit_transform(feat)

		labels = net.blobs['sim'].data
		c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
		     '#ff00ff', '#990000', '#999900', '#009900', '#009999']

		if num_feats == 2:
			for ii in range(0, 10, 1):
				plt.plot(feat[labels == ii, 0].flatten(), feat[labels == ii, 1].flatten(), '.', c=c[ii])
		elif num_feats == 3:
			for ii in range(0, 10, 1):
				xs = new_feats[labels == ii, 0].flatten()
				ys = new_feats[labels == ii, 1].flatten()
				zs = new_feats[labels == ii, 2].flatten()
				ax.scatter(xs, ys, zs, '.', c=c[ii], cmap=plt.cm.Paired)

	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	plt.grid()
	# plt.show()
	plt.savefig(fig_path + features + '.png')
	plt.close()


def plotHistogram(Ds_p, Ds_n, name):
	plt.figure()
	plt.hist(Ds_p, bins=50, histtype='stepfilled', normed=True, color='g',
	         label='x+, mean={:.2f}'.format(np.mean(Ds_p)) + ', std={:.2f}'.format(np.std(Ds_p)))
	plt.hist(Ds_n, bins=50, histtype='stepfilled', normed=True, color='r', alpha=0.5,
	         label='x-, mean={:.2f}'.format(np.mean(Ds_n)) + ', std={:.2f}'.format(np.std(Ds_n)))
	plt.title("Distance Histogram")
	plt.xlabel("Distance")
	plt.ylabel("Instances")
	plt.legend()
	plt.savefig(name)
	plt.close()


def vis_square(data, name):
	"""Take an array of shape (n, height, width) or (n, height, width, 3)
	   and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

	# normalize data for display
	data = (data - data.min()) / (data.max() - data.min())

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0, n ** 2 - data.shape[0]),
	            (0, 1), (0, 1))  # add some space between filters
	           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
	data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

	plt.figure()
	plt.imshow(data)
	plt.axis('off')
	plt.savefig(name)
	plt.close()


def saveFilters(filters, const, total_iter, feat, name):
	path = './data/results/' + name + '_filters_' + const + '_it_' + str(total_iter) + '_' + '.png'
	vis_square(filters[:, 1, :, :], path)

	path = './data/results/' + name + '_features_' + const + '_it_' + str(total_iter) + '_' + '.png'
	vis_square(feat, path)


def plotMeanCurves(MODEL, NET, path_features, path_curves, val_set):
	# val_set = '/home/olaia.artieda/TFM/DATASETS/MNIST/testing/'
	_, test_images, test_labels = load_images(val_set)

	class_AUC = []

	net = caffe.Net(NET, MODEL, caffe.TEST)

	num_images = len(test_images)
	dim_images = test_images[0].shape[1]
	num_feats = net.blobs['ip1'].data.shape[1]
	batch_size = net.blobs['ip1'].data.shape[0]
	num_batches = num_images / batch_size

	aux = num_batches * batch_size
	Features = np.zeros((aux, num_feats), dtype=np.float32)
	Labels = np.zeros((aux, 1), dtype=np.int32)

	for i in range(num_batches):
		net.forward()
		Features[(i * batch_size):(i * batch_size) + batch_size] = net.blobs['ip1'].data
		labels = net.blobs['sim'].data
		Labels[(i * batch_size):(i * batch_size) + batch_size] = labels.reshape(batch_size, 1)

	D = distance.squareform(distance.pdist(Features, 'euclidean'))

	# fig, ax1 = plt.subplots()
	# ax1.imshow(D, extent=[0, aux, 0, aux])
	# plt.title('Feature Matrix')
	# plt.savefig(path_features)
	# plt.close()

	# Class-wise PR curve
	AllClasses_Mean_P = np.zeros((1, aux))
	AllClasses_Mean_R = np.zeros((1, aux))
	AllClasses_Mean_AUC = 0

	Labs = np.unique(Labels)

	plt.figure()
	legend_text = []

	for lb in Labs:
		idxs = find(Labels == lb)
		N = len(idxs)

		GroundTruth = np.zeros(aux)
		GroundTruth[idxs] = 1

		Mean_P = np.zeros(aux)
		Mean_R = np.zeros(aux)
		Mean_AUC = 0

		for img in range(N):
			recall, precision, auc = PR_curve(D[idxs[img], :], GroundTruth)

			Mean_P += precision
			Mean_R += recall
			Mean_AUC += auc

		Mean_P /= float(N)
		Mean_R /= float(N)
		Mean_AUC /= float(N)
		class_AUC.append(Mean_AUC)

		# axes = plt.gca()
		# axes.set_ylim([0, 1.1])
		plt.plot(Mean_R, Mean_P)
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		legend_text.append('Class ' + str(lb) + ': mAP={0:.3f}'.format(Mean_AUC))

		AllClasses_Mean_P += Mean_P
		AllClasses_Mean_R += Mean_R
		AllClasses_Mean_AUC += Mean_AUC

	plt.legend(legend_text, loc=0)
	plt.savefig(path_curves)
	plt.close()

	AllClasses_Mean_P /= float(len(Labs))
	AllClasses_Mean_R /= float(len(Labs))
	AllClasses_Mean_AUC /= float(len(Labs))

	return class_AUC
