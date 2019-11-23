# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2019-11-23 18:25
desc:
'''

from retrieval_feature import *
from retrieval_index import *
from classify import class_results
from PIL import  Image
from PIL import Image, ImageFile, TiffImagePlugin

'''
ImageFile.LOAD_TRUNCATED_IMAGES=True
TiffImagePlugin.READ_LIBTIFF=True
Image.DEBUG=True
'''

def main(img_dir, network, hash_size, input_dim, num_hashtables, feature_path, index_path, out_similar_dir, out_similar_file_dir, all_csv_file, num_results):
	# classify
	class_results(img_dir)

	# extract feature
	AntiFraudFeatureDataset(img_dir, network, feature_path, index_path).constructfeature(hash_size, input_dim, num_hashtables)

	# similar index
	EvaluteMap(out_similar_dir, out_similar_file_dir, all_csv_file, feature_path, index_path).retrieval_images()


if __name__ == "__main__":
	pass





