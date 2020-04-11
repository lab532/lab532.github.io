### 音频处理库librosa的安装
官方文档中给出了非常详细的安装方法
[http://librosa.github.io/librosa/install.html](http://librosa.github.io/librosa/install.html)

### librosa.core.stft的使用
函数声明：
``librosa.core.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=<class 'numpy.complex64'>, pad_mode='reflect')``
常用参数说明：
``y：输入的numpy数组，要求都是实数``
``n_fft：fft的长度，默认2048``
``hop_length：stft中窗函数每次步进的单位``
``win_length：窗函数的长度``
``window：窗函数的类型``
``return：一个1+n_fft/2*1+len(y)/hop_length的二维复数矩阵，其实就是时频谱``
参考：
[http://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa.core.stft](http://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa.core.stft)

### 绘图函数
主要用这两个
``matplotlib.pyplot.pcolormesh()``
``matplotlib.pyplot.colorbar()``

### demo
```
import matplotlib.pyplot as plt
import librosa.core as lc
import numpy as np

fs = 16000
n_fft = 512

f = fs*np.array(range(int(1+n_fft/2)))/(n_fft/2)

path = "xxx.wav"

data = lc.load(path,sr=fs)

length = len(data[0])

spec = np.array(lc.stft(data[0], n_fft=512, hop_length=160, win_length=400, window='hann'))

plt.pcolormesh(np.array(range(int(length/160+1)))/fs, f, np.abs(spec))
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
plt.show()
