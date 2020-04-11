GPU跑模型报错
`RuntimeError: CUDA out of memory. Tried to allocate 38.00 MiB (GPU 0; 10.76 GiB total capacity; 9.71 GiB already allocated; 5.56 MiB free; 9.82 GiB reserved in total by PyTorch)`

* 应该有三个原因
1. GPU还有其他进程占用显存，导致本进程无法分配到足够的显存
2. 缓存过多，使用`torch.cuda.empty_cache()`清理缓存
3. 卡不行，换块显存更大的卡吧

* 除此之外，注意pytorch在test时，一定要加上
```
with torch.no_grad():
  # test process
```
否则会使显存加倍导致OOM错误

* 碰到一个奇怪的事，将训练到一半的模型保存下来，下次训练加载进来，这时加载的时候会占用一部分显存，加载后训练又占用相同显存，结构导致OOM，是我程序写错了吗？
