---
layout:     post
title:      "ubuntu+anaconda+离线pytorch环境搭建"
subtitle:   ""
date:       2020-3-19
author:     "HieDean"
header-img: "img/7.jpg"
tags:
    - 天大毕设
    - anaconda
    - pytorch
    - python
---

### 官网下载anaconda
![](/img/blog/1240)
得到
![](/img/blog/1241)

因为用的是学校的设备，所以要`scp filename hostname@hostIP`上传至服务器
服务器端执行`sh Anaconda3-2020.02-Linux-x86_64.sh`即可

然后还要配置环境变量

`echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc`

`source ~/.bashrc`

### 卸载anaconda
删除整个anaconda3文件即可
`rm -rf anaconda3`

### 下载pytorch之前先查看cuda和cudnn版本
cuda 版本

`cat /usr/local/cuda/version.txt`

或`nvcc -V`

cudnn 版本(命令执行后得到的三个数字即cudnn版本)

`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

（前提是保证机器上已经安装好了cuda和cudnn）

### 离线下载pytorch
参考[https://www.cnblogs.com/darkknightzh/p/12000809.html](https://www.cnblogs.com/darkknightzh/p/12000809.html)

由于使用的是学校的机器，所以先本地下载安装包，再上传

1. conda官网下载pytorch：[https://anaconda.org/pytorch/pytorch/files](https://anaconda.org/pytorch/pytorch/files)(注意cuda和cudnn版本)
![](/img/blog/1242)

2. conda官网下载torchvision：[https://anaconda.org/pytorch/torchvision/files](https://anaconda.org/pytorch/torchvision/files)
![](/img/blog/1243)

3. conda官网下载cudatoolkit：[https://anaconda.org/anaconda/cudatoolkit/files](https://anaconda.org/anaconda/cudatoolkit/files)
![](/img/blog/1244)

下载完成并上传至服务器后

`conda install --offline pytorch-1.3.1-py3.7_cuda10.1.243_cudnn7.6.3_0.tar.bz2`

`conda install --offline torchvision-0.4.2-py37_cu101.tar.bz2`

`conda install --offline cudatoolkit-10.1.243-h6bb024c_0.tar.bz2`

这三大件安装完成后，我按照参考文章中的说法，进行了一次
`conda install pytorch torchvision cudatoolkit=10.1`

结果可以看到图中torchvision（需要回退版本）和cudatoolkit仍然需要下载一遍

![](/img/blog/1245)

可是学校的机器流量不能这么挥霍啊！！！

于是我没有回退torchvision，也没有重新下载cudatoolkit，而是用`conda install`将其他包一个一个下载了下来（建议pillow使用6.1版本）

反正最后能使
![](/img/blog/1246)


