### 安装
`pip install tensorflow`
`pip install tensorboardx`
据说tensorboardx是基于tensorflow的tensorboard的所以要先安装tensorflow
### 简单使用
```
from tensorboardX import SummaryWriter   
writer = SummaryWriter()
# ...
writer.add_scalar('train_loss', loss, n)
writer.add_graph(model, torch.rand([1,3,224,224]))
```
### 可视化结果命令
`tensorboard --logdir=runs`
官方文档[https://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html](https://tensorboardx.readthedocs.io/en/latest/tutorial_zh.html)
