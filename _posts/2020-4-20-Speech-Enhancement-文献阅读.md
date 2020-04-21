---
layout:     post
title:      "Speech Enhancement 文献阅读"
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


文章还介绍了一种自回归滑动平均模型ARMA（atuo-regressive moving average filter），说这玩意儿可以提高低信噪比下语音分离的表现，我打算把它弄到Phasen里试试。


ARMA:


$$C\limits^{\^}(t)=\frac{C\limits^{\^}(t-m)+\cdots +C\limits^{\^}(t)+\cdots +C\limits^{\^}(t+m)}{2m+1}$$

