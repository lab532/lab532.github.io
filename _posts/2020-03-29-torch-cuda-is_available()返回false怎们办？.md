换了台服务器重新搭环境
环境搭好之后发现`torch.cuda.is_available()`返回`false`确实懵了一下

诶，别多想了，一定是版本问题，查去吧
### 查看显卡驱动版本
`nvidia-smi`
### 查看cuda版本
`nvcc -V`
或
`cat /usr/local/cuda/version.txt`
### 查看cudnn版本
`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`
### 查看pytorch版本
`python`
`>>import torch`
`>>torch.__version__`
