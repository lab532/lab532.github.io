---
layout:     post
title:      "文献阅读: Speech Enhancement"
subtitle:   ""
date:       2020-4-20
author:     "HieDean"
header-img: "img/12.jpg"
tags:
    - 天大毕设
---
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

### On Training Targets for Supervised Speech Separation
这篇文章让我大概了解了目前这个邻域的一些技术与做法。


首先要区分语音增强（Speech Enhancement）、语音分离（Speech Separation）、语音解混响（Speech Dereverberation），这三者的目的是有区别的，但使用的方法却很相似。其次，它们其实都是以完成自动语音识别（ASR: automatic speech recognition）为目的的。


这篇文章主要讨论有监督的基于模型（model）的方法，与之相对应的是传统的无监督的基于信号处理（signal processing）的方法。当然，基于model的方法也会借鉴很多信号处理的思想。


基于model的方法又可以分为基于神经网络的方法和基于非负矩阵（NMF: non-negative factorization matrix）的方法。我在另一篇文章还看到过基于隐马尔科夫（HMM）的方法。（以上含有个人见解）


语音的处理从空间上可以分为时域上的和时频域上的。这里所说的时频域是指对一段语音信号进行分帧，并对每一帧做fft得到的二维时频谱，这种方法叫做短时傅里叶变换（STFT），如果我们取STFT时频谱的幅度值，便得到了FFT-MAG，如果取复值，便得到了FFT-MASK。除了STFT，还有一种时频谱经常被使用，叫GF-POW（Gammatone Frequency Power Spectrum）。


文章中还提到目标（target），也就是我们最终想要获得什么？一是获得一个mask，这是该文讨论的重点，二是获得一个频谱包络（spectral envelope）。


什么是maske呢？mask主要用于时频域，如果带噪的语音的stft时频谱为$X$，干净的语音的时频谱为$Y$，mask为$M$，那么便有$Y=X \cdot M$，从此公式也可以看出，mask同带噪语音以及干净语音的时频谱的大小应该是一样的。


当然，mask也分很多种，最简单的是IBM（ideal binary mask），即每个unit的值不是0就是1，这种最简单的mask把语音增强看作一个二值化问题，或者说二元分类问题：计算带噪时频谱的每一个单元的信噪比，通过比较该单元信噪比与 local criterion 的关系来决定该位置处的mask应为1或是0，这样的mask可以将时频谱中噪声主导的单元置为0，语音主导的单元保留，从而达到降噪的目的。有一种与IBM非常相似的叫TBM（Target Binary Mask），它与IBM的不同点在于信噪比的计算方式，TBM使用的噪声并非实际噪声，而是SSN: speech-shaped noise。


IBM:


$$
IBM(t,f)= \left\{
            \begin{aligned}
            1 \ , \ if SNR(t,f) > LC \\
            0 \ , \ otherwise.
            \end{aligned}
            \right.
$$


IBM的进化版IRM（ideal ratio mask），它与IBM的不同在于mask使用的概率，也就是说不论是语音主导的单元还是噪声主导的单元，mask对其的作用都是衰减或者增益，而不是绝对的保留或置0，用脚趾头也能想到这玩意儿的效果会比IBM好。


IRM:


$$
IRM(t,f)=(\frac{S^2(t,f)}{S^2(t,f)+N^2(t,f)})^{\beta}
$$


文章还简单介绍了一些关于语音信号的feature，比如AMS: amplitude modulation spectrogram、MFCC: mel-frequency cepstral coefficients、GF: 64-channels Gammatone filterbank power spectra


文章还介绍了一种自回归滑动平均模型ARMA（atuo-regressive moving average filter），说这玩意儿可以提高低信噪比时语音分离的表现，我打算把它弄到Phasen里试试。


ARMA:


$$\hat{C}(t)=\frac{\hat{C}(t-m)+\cdots +\mathop{C}(t)+\cdots +\mathop{C}(t+m)}{2m+1}$$


### Singing Voice Separation With Deep U-net Convolutional Networks
这篇文章主要是用两个U-net，一个用于提取一首歌中的旋律，另一个用于提取伴奏，还是基于fft-mask的方法，挺好理解的。


### Binary and ratio time-frequency masks for robust speech recognition
这篇文章看的很艰难，可能是我英文不够好，也可能是太瞌睡了

从题目可以看出这篇文章主要是对比binary mask和ratio mask进行对比。这里又要区分一下ideal和estimated，即binary mask有ideal binary mask和estimated binary mask之分，ratio mask也有ideal ratio mask和estimated ratio mask之分。但我不太清楚ideal和estimated之间的本质区别在哪里，个人理解就是理想的mask是由最开始的一套固定的算法计算得出的，之后出现的其他的计算方法均称为estimated


这里用其他方法估计mask：使用CASA（computational auditory scene analysis）估计IBM；使用一种和ITD（interaural time difference）、IID（interaural intensity difference）有关的信噪比计算IRM。得出的IBM和IRM分别送入数据缺失识别器和传统语音识别器。（CASA也与ITD和IID有关）


传统的语音识别器：基于隐马尔科夫和梅尔倒谱系数的自动语音识别.


数据缺失识别器（missing-data recognizer）：
> This method is based on distinguishing between reliable and unreliable data When speech is contaminated by additive noise some time-frequency units contain predominantly speech energy (reliable) and the rest are dominated by noise energy. The missing-data method treats the latter T-F units as missing or unreliable during recognition. Missing T-F units are identified by thresholding the t-F units based on local SNR. 


所以整个系统就是，单通道信号分别送入不同的HRTF（head-related transfer functions，头部相关传输函数，是一种音效定位算法）得到双耳信号（左耳信号和右耳信号），双耳信号经过stft后又经过两种不同的方法分别得出IBM和IRM，IBM送入数据缺失识别器，IRM送入传统语音识别器。


### Complex Ratio Masking for Monaural Speech Separation
其实看这篇文章之前已经大概知道它要干啥了，所以就不是很费劲。


前面提到mask的发展从IBM到IRM，但是这些mask都是实数，它们仅能体现时频谱的幅度信息。但有研究发现，相位信息的重构对语音增强后的语音的听觉感受和可懂度都非常重要，所以作者就提出了使用DNN去估计cIRM，也就是mask的每一个单元都是一个复数，这就和phasen非常像了

### Phasen-Sensitive and Recongnition-Boosted Speech Separation Using Deep Recurrent Neural Networks
瞅瞅这题目，又是相位敏感、又是识别增强、又是语音分离、又是深度循环神经网络，这到底是个啥？


前面已经说到mask发展到了cIRM，这篇文章也是提出使用复数域的mask，不过和cIRM不同，它的特色在于相位敏感（就是说新发明了一种mask的意思吧我猜）：


$$a^{psf}=\frac{|s|}{|y|}cos(\theta)$$
$$\theta = \theta ^s - \theta ^y$$


这篇文章也提出了使用RNN来提取语音在时间前后上的相关性，并指出要长时间域，不能短时间域，且为了避免gradient vanish和gradient blows up，所以用了LSTM。


这篇文章我觉得牛的一点是，作者认为语音识别和语音分离是一个先有鸡还是先有蛋的问题，语音识别的结果可以提升语音分离的表现能力，语音分离也可以提升语音识别的准确率。因此作者打算把二者结合起来设计一个多任务的前馈系统，既可以识别，也可以分离（虽然和GAN不一样，但这让我想起了GAN）。


但这个系统结构是啥样的，作者没有附图，只是文字描述了一下，我没怎么读懂......

持续更新......
