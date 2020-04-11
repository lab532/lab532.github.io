可以使用pynvml库来查看
### pynvml的安装
`pip install nvidia-ml-py3`
(尝试了一下，`conda install`似乎行不通)
### pynvml的使用
```
import pynvml
pynvml.nvmlInit()
# GPU的id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used)
```
pynvml文档：[https://docs.nvidia.com/deploy/nvml-api/index.html](https://docs.nvidia.com/deploy/nvml-api/index.html)
