---
layout:     post
title:      "pytorch中的tensorboardx安装及简单使用"
subtitle:   ""
date:       2020-3-30
author:     "HieDean"
header-img: "img/9.jpg"
tags:
    - 天大毕设
    - pytorch
---
### 安装
`pip install tensorflow`

`pip install tensorboardx`

据说tensorboardx是基于tensorflow的tensorboard的所以要先安装tensorflow

### 简单使用
```python
from tensorboardX import SummaryWriter   
writer = SummaryWriter()
# ...
writer.add_scalar('train_loss', loss, iteration)
writer.add_graph(model, torch.rand(input.shape))
```

### 查看结果命令
命令行输入

`tensorboard --logdir=runs`

在浏览器输入命令行出现的url即可


### 如果tensorboard的log文件都在远程服务器上，如何在本地访问呢？
首先，在ssh连接时建立ssh隧道，实现远程端口到本地端口的转发。
``ssh -L 16006:127.0.0.1:6006 account@server.address``


具体来说就是将远程服务器的6006端口（tensorboard默认将数据放在6006端口）转发到本地的16006端口，在本地对16006端口的访问即是对远程6006端口的访问，当然，转发到本地某一端口不是限定的，可自由选择。


在远端服务器上开启tensorboard
``tensorboard --logdir=runs``


最后本地访问`http://127.0.0.1:16006/`

官方文档
[https://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html](https://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html)
