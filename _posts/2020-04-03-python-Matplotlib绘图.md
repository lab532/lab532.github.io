```
import matplotlib.pyplot as plt
import numpy as np
import librosa.core as lc

noisy_path = './datasets/noisy_testset_wav/'
clean_path = './datasets/clean_testset_wav/'
wave_name = 'p257_059.wav'
# 准备x轴和y轴
x_axis = np.array(range(int(len(clean_wave)/160+1)))/fs
y_axis = fs*np.array(range(int(1+n_fft/2)))/(n_fft/2)
# 载入数据
noisy_wave = lc.load(noisy_path+wave_name,sr=16000)[0]
clean_wave = lc.load(clean_path+wave_name,sr=16000)[0]
# stft变换
noisy_spec = lc.stft(noisy_wave, n_fft=512, hop_length=160, win_length=400, window='hann')
clean_spec = lc.stft(clean_wave, n_fft=512, hop_length=160, win_length=400, window='hann')
# 定义一个figure，通俗讲就是一个窗口
figure = plt.figure() 
# 在这一个窗口中添加子图
axe_1 = figure.add_subplot(221)
axe_2 = figure.add_subplot(222)
axe_3 = figure.add_subplot(223)
axe_4 = figure.add_subplot(224)
# 给子图添加名称并绘制
axe_1.set_title('noisy_spec')
temp = axe_1.pcolormesh(y_axis, x_axis, np.abs(noisy_spec)) # 绘图
figure.colorbar(temp, ax=axe_1) # 给子图添加颜色渐变条

axe_2.set_title('clean_spec')
temp = axe_2.pcolormesh(y_axis, x_axis, np.abs(clean_spec))
figure.colorbar(temp, ax=axe_2)

axe_3.set_title('noisy_wave')
axe_3.plot(np.arange(len(noisy_wave))/fs, noisy_wave)

axe_4.set_title('noisy_wave')
axe_4.plot(np.arange(len(clean_wave))/fs, clean_wave)

plt.tight_layout() # 设置布局
plt.show() # 显示
```
