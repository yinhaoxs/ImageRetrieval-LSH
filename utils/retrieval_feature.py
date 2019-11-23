# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2019-11-23 18:26
desc:
'''

import argparse
import os
import time
import pickle
import pdb
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook
from scipy.cluster.vq import *
from lshash.lshash import LSHash
import math
from sklearn.externals import joblib
from classify import class_results

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime


# setting up the visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class ImageProcess():
	def __init__(self, img_dir):
		self.img_dir = img_dir

	def process(self):
		imgs = list()
		for root, dirs, files in os.walk(self.img_dir):
			for file in files:
				img_path = os.path.join(root + os.sep, file)
				try:
					image = Image.open(img_path)
					if max(image.size) / min(image.size) < 5:
						imgs.append(img_path)
					else:
						continue
				except:
					print("image height/width ratio is small")

		return imgs


class AntiFraudFeatureDataset():
	def __init__(self, img_dir, network, feature_path, index_path):
		self.img_dir = img_dir
		self.network = network
		self.feature_path = feature_path
		self.index_path = index_path

	def constructfeature(self, hash_size, input_dim, num_hashtables):
		multiscale = '[1]'
		print(">> Loading network:\n>>>> '{}'".format(self.network))
		# state = load_url(PRETRAINED[args.network], model_dir=os.path.join(get_data_root(), 'networks'))
		state = torch.load(self.network)
		# parsing net params from meta
		# architecture, pooling, mean, std required
		# the rest has default values, in case that is doesnt exist
		net_params = {}
		net_params['architecture'] = state['meta']['architecture']
		net_params['pooling'] = state['meta']['pooling']
		net_params['local_whitening'] = state['meta'].get('local_whitening', False)
		net_params['regional'] = state['meta'].get('regional', False)
		net_params['whitening'] = state['meta'].get('whitening', False)
		net_params['mean'] = state['meta']['mean']
		net_params['std'] = state['meta']['std']
		net_params['pretrained'] = False
		# network initialization
		net = init_network(net_params)
		net.load_state_dict(state['state_dict'])
		print(">>>> loaded network: ")
		print(net.meta_repr())
		# setting up the multi-scale parameters
		ms = list(eval(multiscale))
		print(">>>> Evaluating scales: {}".format(ms))
		# moving network to gpu and eval mode
		net.cuda()
		net.eval()

		# set up the transform
		normalize = transforms.Normalize(
			mean=net.meta['mean'],
			std=net.meta['std']
		)
		transform = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])

		# extract database and query vectors
		print('>> database images...')
		images = ImageProcess(self.img_dir).process()
		vecs, img_paths = extract_vectors(net, images, 224, transforms, ms=ms)
		feature_dict = dict(zip(img_paths, list(vecs.detach().cpu().numpy().T)))
		# index
		lsh = LSHash(hash_size=int(hash_size), input_dim=int(input_dim), num_hashtables=int(num_hashtables))
		for img_path, vec in feature_dict.items():
			lsh.index(vec.flatten(), extra_data=img_path)

		## 保存索引模型
		with open(self.feature_path, "wb") as f:
			pickle.dump(feature_dict, f)
		with open(self.index_path, "wb") as f:
			pickle.dump(lsh, f)

		print("extract feature is done")


if __name__ == '__main__':
	pass

