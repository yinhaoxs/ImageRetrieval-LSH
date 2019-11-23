# 图像检索模型

## 1.模型介绍
    --技术：深度卷积神经网络技术、LSH局部敏感哈希算法、flask web端部署

## 2.预训练模型
    --图像检索预训练模型：https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ

## 3.数据集
    --face recognition数据集：LFW人脸识别数据集（官网提供下载）
      --相同人脸随机构成pairs对，label(y)均设置为0.（同一人的pairs对的风险值相同）
    --KYD风险人脸数据集：采集带有风险值（label）的图像数据集
      --随机构成人脸pairs对，label（y）为风险值的差值.

## 4.数据预处理
    数据大小处理为（112*112）
    --LFW数据集：同一人的不同图片构成pairs对，写入csv文件便于后续训练
    --KYD数据集：所有训练数据随机构成pairs对，无需处理，训练时batch中会进行随机配对
    
## 5.模型训练
    --batch训练：读取LFW文件，每次随机取64对图像输入到batch；随机在dataloader中取64对KYD图片对输入到batch（batch=12），确定训练时两组不同          的数据比例为1：1，保证训练出的模型gini高，方差低
    --loss函数:采用soft_bce loss函数作为模型收敛的策略函数
    --分布式训练 python train.py
    --单卡测试 python test.py
    --demo
    
    
## 6.指标
    --gini系数
    --同一人不用人脸的风险系数方差std
     代码地址：git clone https://github.com/yinhaoxss/Arcface_RankNet.git

  
