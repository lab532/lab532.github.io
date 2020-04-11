### 安装
`pip install tensorflow`

`pip install tensorboardx`

据说tensorboardx是基于tensorflow的tensorboard的所以要先安装tensorflow

### 简单使用
```
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

官方文档
[https://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html](https://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html)
