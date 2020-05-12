# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs
data: 2019-11-24 00:13
desc:
'''

import requests
import time
import hashlib
import sys
import os, torch

URL = "http://0.0.0.0：15788/images_retrieval"


def run_get():
    r = requests.get(URL)
    print(r.text)


def run_post():
    img_path = "./images/1.jpg"
    q_no = ""
    q_did = ""
    q_id = ""
    type = ""

    print(img_path)
    img_file = open(img_path, "rb")
    files = {'query': open(img_path, "rb")}
    data = {'sig': hashlib.md5(img_file.read()).hexdigest(), 'q_no': "", 'q_did': "", 'q_id': "", 'type': ""}

    t = time.time()
    r = requests.post(URL, data=data, files=files)
    print(r)
    print(r.text)
    print(r.content)
    print("*" * 100)
    res = r.json()
    print("post time used: {}".format(time.time() - t))
    print(res)

    if res['err'] == 0:
        print("query image:{}".format(res['data']['query']))
        print(res['data'])
        print("----------results----------")
        for x in res['data']['results']:
            print("x:", x)
            print("s_no: {}, s_did: {}, s_id: {}".format(x['s_no'], x['s_did'], x['s_id']))

    else:
        print(res['err'])
        print(res['msg'])


if __name__ == '__main__':
    run_get()
    run_post()
