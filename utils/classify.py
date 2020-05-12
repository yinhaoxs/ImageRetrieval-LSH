# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs
data: 2019-11-23 18:29
desc:
'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import time
from collections import OrderedDict

# config.py
BATCH_SIZE = 16
PROPOSAL_NUM = 6
CAT_NUM = 4
INPUT_SIZE = (448, 448)  # (w, h)
DROP_OUT = 0.5
CLASS_NUM = 37


# resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature1 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout(p=0.5)(x)
        feature2 = x
        x = self.fc(x)

        return x, feature1, feature2


# model.py
class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)


class AttentionNet(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        self.pretrained_model = ResNet(Bottleneck, [3, 4, 6, 3])
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), 200)
        self.partcls_net = nn.Linear(512 * 4, 200)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)

    def forward(self, x):
        resnet_out, rpn_feature, feature = self.pretrained_model(x)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_net(rpn_feature.detach())
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        # concat_logits have the shape: B*200
        concat_out = torch.cat([part_feature, feature], dim=1)
        concat_logits = self.concat_net(concat_out)
        raw_logits = resnet_out
        # part_logits have the shape: B*N*200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]


def list_loss(logits, targets):
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
    return torch.stack(loss)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    loss = Variable(torch.zeros(1).cuda())
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


# anchors.py
_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)


def generate_default_anchor_maps(anchors_setting=None, input_shape=INPUT_SIZE):
    """
    generate default anchor
    :param anchors_setting: all informations of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:

        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5

                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

    return center_anchors, edge_anchors, anchor_areas


def hard_nms(cdds, topn=10, iou_thresh=0.25):
    if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
        raise TypeError('edge_box_map should be N * 5+ ndarray')

    cdds = cdds.copy()
    indices = np.argsort(cdds[:, 0])
    cdds = cdds[indices]
    cdd_results = []

    res = cdds

    while res.any():
        cdd = res[-1]
        cdd_results.append(cdd)
        if len(cdd_results) == topn:
            return np.array(cdd_results)
        res = res[:-1]

        start_max = np.maximum(res[:, 1:3], cdd[1:3])
        end_min = np.minimum(res[:, 3:5], cdd[3:5])
        lengths = end_min - start_max
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (
                cdd[4] - cdd[2]) - intersec_map)
        res = res[iou_map_cur < iou_thresh]

    return np.array(cdd_results)


#### -------------------------------如何定义batch的读写方式-------------------------------
# 默认读写方式
def default_loader(path):
    try:
        img = Image.open(path).convert("RGB")
        if img is not None:
            return img
    except:
        print("error image:{}".format(path))


def opencv_isvalid(img_path):
    img_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr


# 判断图片是否为无效
def IsValidImage(img_path):
    vaild = True
    if img_path.endswith(".tif") or img_path.endswith(".tiff"):
        vaild = False
        return vaild
    try:
        img = opencv_isvalid(img_path)
        if img is None:
            vaild = False
        return vaild
    except:
        vaild = False
        return vaild


class MyDataset(Dataset):
    def __init__(self, dir_path, transform=None, loader=default_loader):
        fh, imgs = list(), list()
        num = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                try:
                    img_path = os.path.join(root + os.sep, file)
                    num += 1
                    if IsValidImage(img_path):
                        fh.append(img_path)
                    else:
                        os.remove(img_path)

                except:
                    print("image is broken")
        print("total images is:{}".format(num))

        for line in fh:
            line = line.strip()
            imgs.append(line)

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        fh = self.imgs[item]
        img = self.loader(fh)
        if self.transform is not None:
            img = self.transform(img)
        return fh, img

    def __len__(self):
        return len(self.imgs)


#### -------------------------------如何定义batch的读写方式-------------------------------


#### -------------------------------图像模糊的定义-------------------------------
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64f).var()


## 如何定义接口函数
def imgQualJudge(img, QA_THRESHOLD):
    '''
    :param img:
    :param QA_THRESHOLD: 越高越清晰
    :return: 是否模糊，0为模糊，1为清晰
    '''

    norheight = 1707
    norwidth = 1280
    flag = 0
    # 筛选尺寸
    if max(img.shape[0], img.shape[1]) < 320:
        flag = '10002'
        return flag

    # 模糊筛选部分
    if img.shape[0] <= img.shape[1]:
        size1 = (norheight, norwidth)
        timage = cv2.resize(img, size1)
    else:
        size2 = (norwidth, norheight)
        timage = cv2.resize(img, size2)

    tgray = cv2.cvtColor(timage, cv2.COLOR_BGR2GRAY)
    halfgray = tgray[0:int(tgray.shape[0] / 2), 0:tgray.shape[1]]
    norgrayImg = np.zeros(halfgray.shape, np.int8)
    cv2.normalize(halfgray, norgrayImg, 0, 255, cv2.NORM_MINMAX)
    fm = variance_of_laplacian(norgrayImg)  # 模糊值
    if fm < QA_THRESHOLD:
        flag = '10001'
        return flag
    return flag


def process(img_path):
    img = Image.open(img_path).convert("RGB")
    valid = True
    low_quality = "10001"
    size_error = "10002"

    flag = imgQualJudge(np.array(img), 5)
    if flag == low_quality or flag == size_error or not img or 0 in np.asarray(img).shape[:2]:
        valid = False

    return valid


#### -------------------------------图像模糊的定义-------------------------------

def build_dict():
    dict_club = dict()
    dict_club[0] = ["身份证", 0.999999]
    dict_club[1] = ["校园卡", 0.890876]
    return dict_club


class Classifier():
    def __init__(self):
        self.device = torch.device('cuda')
        self.class_id_name_dict = build_dict()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = 448
        self.use_cuda = torch.cuda.is_available()
        self.model = AttentionNet(topN=4)
        self.model.eval()

        checkpoint = torch.load("./.ckpt")
        newweights = checkpoint['net_state_dict']

        # 多卡测试转为单卡
        new_state_dic = OrderedDict()
        for k, v in newweights.items():
            name = k[7:] if k.startwith("module.") else k
            new_state_dic[name] = v

        self.model.load_state_dict(new_state_dic)
        self.model = self.model.to(self.device)

    def evalute(self, dir_path):
        data = MyDataset(dir_path, transform=self.preprocess)
        dataloader = DataLoader(dataset=data, batch_size=32, num_workers=8)

        self.model.eval()
        with torch.no_grad():
            num = 0
            for i, (data, path) in enumerate(dataloader, 1):
                data = data.to(self.device)
                output = self.model(data)
                for j in range(len(data)):
                    img_path = path[j]
                    img_output = output[1][j]
                    score, label, type = self.postprocess(img_output)
                    out_dict, score = self.process(score, label, type)
                    class_id = out_dict["results"]["class2"]["code"]
                    num += 1
                    if class_id != '00038':
                        os.remove(img_path)
                    else:
                        continue

    def preprocess(self, img):
        img = transforms.Resize((600, 600), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(self.mean, self.std)

    def postprocess(self, output):
        pred_logits = F.softmax(output, dim=0)
        score, label = pred_logits.max(0)
        score = score.item()
        label = label.item()
        type = self.class_id_name_dict[label][0]
        return score, label, type

    def process(self, score, label, type):
        success_code = "200"
        lower_conf_code = "10008"

        threshold = float(self.class_id_name_dict[label][1])
        if threshold > 0.99:
            threshold = 0.99
        if threshold < 0.9:
            threshold = 0.9
        ## 设置查勘图片较低的阈值
        if label == 38:
            threshold = 0.5

        if score > threshold:
            status_code = success_code
            pred_label = str(label).zfill(5)
            print("pred_label:", pred_label)
            return {"code:": status_code, "message": '图像分类成功',
                    "results": {"class2": {'code': pred_label, 'name': type}}}, score
        else:
            status_code = lower_conf_code
            return {"code:": status_code, "message": '图像分类置信度低，不返回结果',
                    "results": {"class2": {'code': '', 'name': ''}}}, score


def class_results(img_dir):
    Classifier().evalute(img_dir)


if __name__ == "__main__":
    pass
