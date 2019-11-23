# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2019-11-23 21:51
desc:
'''
import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime
from cirtorch.networks.imageretrievalnet_cpu import init_network, extract_vectors
from cirtorch.datasets.datahelpers import imresize

from PIL import Image
import numpy as np
import pandas as pd
from flask import Flask, request
import json, io, sys, time, traceback, argparse, logging, subprocess, pickle, os, yaml,shutil
import cv2
import pdb
from werkzeug.utils import cached_property
from apscheduler.schedulers.background import BackgroundScheduler
from multiprocessing import Pool

app = Flask(__name__)

@app.route("/")
def index():
	return ""

@app.route("/images/*", methods=['GET','POST'])
def accInsurance():
	"""
	flask request process handle
	:return:
	"""
	try:
		if request.method == 'GET':
			return json.dumps({'err': 1, 'msg': 'POST only'})
		else:
			app.logger.debug("print headers------")
			headers = request.headers
			headers_info = ""
			for k, v in headers.items():
				headers_info += "{}: {}\n".format(k, v)
			app.logger.debug(headers_info)

			app.logger.debug("print forms------")
			forms_info = ""
			for k, v in request.form.items():
				forms_info += "{}: {}\n".format(k, v)
			app.logger.debug(forms_info)

			if 'query' not in request.files:
				return json.dumps({'err': 2, 'msg': 'query image is empty'})

			if 'sig' not in request.form:
				return json.dumps({'err': 3, 'msg': 'sig is empty'})

			if 'q_no' not in request.form:
				return json.dumps({'err': 4, 'msg': 'no is empty'})

			if 'q_did' not in request.form:
				return json.dumps({'err': 5, 'msg': 'did is empty'})

			if 'q_id' not in request.form:
				return json.dumps({'err': 6, 'msg': 'id is empty'})

			if 'type' not in request.form:
				return json.dumps({'err': 7, 'msg': 'type is empty'})

			img_name = request.files['query'].filename
			img_bytes = request.files['query'].read()
			img = request.files['query']
			sig = request.form['sig']
			q_no = request.form['q_no']
			q_did = request.form['q_did']
			q_id = request.form['q_id']
			type = request.form['type']

			if str(type) not in types:
				return json.dumps({'err': 8, 'msg': 'type is not exist'})

			if img_bytes is None:
				return json.dumps({'err': 10, 'msg': 'img is none'})

			results = imageRetrieval().retrieval_online_v0(img, q_no, q_did, q_id, type)

			data = dict()
			data['query'] = img_name
			data['sig'] = sig
			data['type'] = type
			data['q_no'] = q_no
			data['q_did'] = q_did
			data['q_id'] = q_id
			data['results'] = results

			return json.dumps({'err': 0, 'msg': 'success', 'data': data})

	except:
		app.logger.exception(sys.exc_info())
		return json.dumps({'err': 9, 'msg': 'unknow error'})


class imageRetrieval():
	def __init__(self):
		pass

	def cosine_dist(self, x, y):
		return 100 * float(np.dot(x, y))/(np.dot(x,x)*np.dot(y,y)) ** 0.5

	def inference(self, img):
		try:
			input = Image.open(img).convert("RGB")
			input = imresize(input, 224)
			input = transforms(input).unsqueeze()
			with torch.no_grad():
				vect = net(input)
			return vect
		except:
			print('cannot indentify error')

	def retrieval_online_v0(self, img, q_no, q_did, q_id, type):
		# load model
		query_vect = self.inference(img)
		query_vect = list(query_vect.detach().numpy().T[0])

		lsh = lsh_dict[str(type)]
		response = lsh.query(query_vect, num_results=1, distance_func = "cosine")

		try:
			similar_path = response[0][0][1]
			score = np.rint(self.cosine_dist(list(query_vect), list(response[0][0][0])))
			rank_list = similar_path.split("/")
			s_id, s_did, s_no = rank_list[-1].split("_")[-1].split(".")[0], rank_list[-1].split("_")[0], rank_list[-2]
			results = [{"s_no": s_no, "r_did": s_did, "s_id": s_id, "score": score}]
		except:
			results = []

		img_path = "/{}/{}_{}".format(q_no, q_did, q_id)
		lsh.index(query_vect, extra_data=img_path)
		lsh_dict[str(type)] = lsh

		return results



class initModel():
	def __init__(self):
		pass

	def init_model(self, network, model_dir, types):
		print(">> Loading network:\n>>>> '{}'".format(network))
		# state = load_url(PRETRAINED[args.network], model_dir=os.path.join(get_data_root(), 'networks'))
		state = torch.load(network)
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
		# moving network to gpu and eval mode
		# net.cuda()
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

		lsh_dict = dict()
		for type in types:
			with open(os.path.join(model_dir, "dataset_index_{}.pkl".format(str(type))), "rb") as f:
				lsh = pickle.load(f)

			lsh_dict[str(type)] = lsh

		return net, lsh_dict, transforms

	def init(self):
		with open('config.yaml', 'r') as f:
			conf = yaml.load(f)

		app.logger.info(conf)
		host = conf['website']['host']
		port = conf['website']['port']
		network = conf['model']['network']
		model_dir = conf['model']['model_dir']
		types = conf['model']['type']

		net, lsh_dict, transforms = self.init_model(network, model_dir, types)

		return host, port, net, lsh_dict, transforms, model_dir, types


def job():
	for type in types:
		with open(os.path.join(model_dir, "dataset_index_{}_v0.pkl".format(str(type))), "wb") as f:
			pickle.dump(lsh_dict[str(type)], f)
			

if __name__ == "__main__":
	"""
	start app from ssh
	"""
	scheduler = BackgroundScheduler()
	host, port, net, lsh_dict, transforms, model_dir, types = initModel().init()
	app.run(host=host, port=port, debug=True)
	print("start server {}:{}".format(host, port))

	scheduler.add_job(job, 'interval', seconds= 30)
	scheduler.start()

else:
	"""
	start app from gunicorn
	"""
	scheduler = BackgroundScheduler()
	gunicorn_logger = logging.getLogger("gunicorn.error")
	app.logger.handlers = gunicorn_logger.handlers
	app.logger.setLevel(gunicorn_logger.level)

	host, port, net, lsh_dict, transforms, model_dir, types = initModel().init()
	app.logger.info("started from gunicorn...")

	scheduler.add_job(job, 'interval', seconds=30)
	scheduler.start()



