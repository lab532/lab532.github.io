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


语音的处理从空间上可以分为时域上的和时频域上的。这里所说的时频域是指对一段语音信号进行分帧，并对每一帧做fft得到的二维时频谱。


文章中还提到目标（target），也就是我们最终想要获得什么？一是获得一个mask，这是该文讨论的重点，二是获得一个频谱包络（spectral envelope）。


什么是maske呢？mask主要用于时频域，如果带噪的语音的stft时频谱为$X$，干净的语音的时频谱为$Y$，mask为$M$，那么便有$Y=X \cdot M$。当然，mask也分很多种，最简单的是IBM（ideal binary mask），即
