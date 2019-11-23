# 图像检索模型

## 1.模型介绍
    --技术：深度卷积神经网络技术、LSH局部敏感哈希算法、flask web端部署

## 2.预训练模型
    --图像检索预训练模型：http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth

## 3.数据集
    --训练数据集：针对业务数据进行迁移学习
    --生产上数据集查重
   
## 4.数据预处理
    数据大小处理为（224*224）
    --筛选出tiff、tif等格式文件，并解决pillow的底层问题（opencv解决conver（“RGB”）问题）
    --数据分类筛选：采用nts网络进行细粒度分类
    
## 5.模型业务使用
    --分类：python classify.py
    --特征提取：python retrieval_feature.py
    --图像离线检索：python retrieval_index.py
    --在线部署：python interface.py (采用flask框架部署，同时离线更新数据库)
       
## 6.指标
    --map：0.93
    

  
